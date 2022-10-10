// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion.h"
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include "llvm/Support/SourceMgr.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/tool_funcs.h"
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"
#include "pytorch_blade/compiler/mlir/converters/torch_mlir_op_filter.h"

#include "function_importer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

std::string GenerateMlirModuleString(bool pretty, mlir::ModuleOp module) {
  std::string s;
  ::llvm::raw_string_ostream ss(s);
  mlir::OpPrintingFlags print_flags;
  // IR generated with 'prettyForm' is not parsable.
  if (pretty) {
    print_flags.elideLargeElementsAttrs();
  }
  print_flags.enableDebugInfo(/*prettyForm*/ pretty);
  module.print(ss, print_flags);
  return ss.str();
}

std::string GetAttrString(const ::llvm::ArrayRef<std::string>& str_vec) {
  std::string s;
  ::llvm::raw_string_ostream ss(s);
  ::llvm::interleave(str_vec, ss, ",");
  return ss.str();
}

std::vector<std::string> GetDeviceStrs(
    at::ArrayRef<const torch::jit::Value*> vals) {
  std::vector<std::string> dev_strs;
  dev_strs.reserve(vals.size());
  for (auto& val : vals) {
    // default to device cpu
    if (is_gpu_tensor_type(*val)) {
      dev_strs.push_back("gpu");
    } else {
      dev_strs.push_back("cpu");
    }
  }
  return dev_strs;
}

std::vector<std::string> GetDebugNames(
    at::ArrayRef<const torch::jit::Value*> vals) {
  std::vector<std::string> debug_names;
  debug_names.reserve(vals.size());
  for (auto& val : vals) {
    debug_names.push_back(val->debugName());
  }
  return debug_names;
}

mlir::NamedAttribute GetFunctionAttrs(
    mlir::Builder& builder,
    at::ArrayRef<const torch::jit::Value*> inputs,
    at::ArrayRef<const torch::jit::Value*> outputs) {
  auto input_devices = GetDeviceStrs(inputs);
  auto output_devices = GetDeviceStrs(outputs);
  auto input_names = GetDebugNames(inputs);
  auto output_names = GetDebugNames(outputs);

  auto inputs_attr = builder.getNamedAttr(
      "inputs", builder.getStringAttr(GetAttrString(input_names)));
  auto outputs_attr = builder.getNamedAttr(
      "outputs", builder.getStringAttr(GetAttrString(output_names)));
  auto input_dev_str = GetAttrString(input_devices);
  auto input_placements_attr = builder.getNamedAttr(
      "input_placements", builder.getStringAttr(input_dev_str));
  auto output_dev_str = GetAttrString(output_devices);
  auto output_placements_attr = builder.getNamedAttr(
      "output_placements", builder.getStringAttr(output_dev_str));

  return builder.getNamedAttr(
      "tf.entry_function",
      builder.getDictionaryAttr(
          {inputs_attr,
           outputs_attr,
           input_placements_attr,
           output_placements_attr}));
}

std::tuple<mlir::func::FuncOp, std::string, std::string> CreateMlirFunction(
    MhloConversionContext& ctx,
    const std::string& function_name,
    at::ArrayRef<const torch::jit::Value*> inputs,
    at::ArrayRef<const torch::jit::Value*> outputs) {
  SmallVec4<mlir::Type> args;
  SmallVec4<mlir::Type> rets;

  auto& builder = *ctx.builder;
  for (auto& input : inputs) {
    auto mlir_tensor_type = BuildMlirRankedTensorType(builder, *input);
    args.emplace_back(mlir_tensor_type);
  }

  for (auto& output : outputs) {
    // The output type would be reset during the function building being
    // finalized. Currently it's set to placeholder unranked tensor type.
    auto unk_default_type = mlir::UnrankedTensorType::get(builder.getF32Type());
    rets.emplace_back(unk_default_type);
  }

  auto mlir_context = ctx.mlir_module->getContext();
  auto mlir_func_type = mlir::FunctionType::get(mlir_context, args, rets);
  auto entry_attr = GetFunctionAttrs(builder, inputs, outputs);
  auto func = builder.create<mlir::func::FuncOp>(
      ctx.mlir_module->getLoc(),
      function_name,
      mlir_func_type,
      ::llvm::ArrayRef<::mlir::NamedAttribute>{entry_attr});

  auto entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  size_t idx = 0;
  for (auto& input : inputs) {
    ctx.value_map[input] = func.getArgument(idx++);
  }

  return std::make_tuple(
      func,
      GetAttrString(GetDeviceStrs(inputs)),
      GetAttrString(GetDeviceStrs(outputs)));
}

// TODO(wenyi): Using registration system instead of using explicit listing.
bool SafeToCallConverter(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  if (!schema)
    return false;
  auto& name = schema->operator_name().name;
  return name == "aten::_convolution";
}

inline bool IsNonTensorOrTypeAnalyzed(const c10::TypePtr type) {
  TORCH_CHECK(type != nullptr);
  auto tensor_type = type->cast<c10::TensorType>();
  if (tensor_type == nullptr) {
    // there is no need to check any type other than TensorType
    return true;
  }
  return tensor_type->scalarType() && tensor_type->device() &&
      tensor_type->dim();
}

inline bool AllTensorTypeAnalyzed(const torch::jit::Node& node) {
  for (auto inp : node.inputs()) {
    if (!IsNonTensorOrTypeAnalyzed(inp->type())) {
      return false;
    }
  }

  for (auto out : node.outputs()) {
    if (!IsNonTensorOrTypeAnalyzed(out->type())) {
      return false;
    }
  }
  return true;
}

void RegisterDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
}

bool IsMlirMhloSupported(const torch::jit::Node& node) {
  if (IsTorchMlirAvailable()) {
    if (!node.kind().is_prim() && !AllTensorTypeAnalyzed(node)) {
      return false;
    }
    auto supported = IsTorchMlirSupported(node);
    bool enable_printing =
        env::ReadBoolFromEnvVar("TORCH_BLADE_MHLO_DEBUG_LOG", false);

    if (enable_printing &&
        !(supported || prim::FusionGroup == node.kind() ||
          prim::Param == node.kind())) {
      LOG(WARNING) << "Not in white list: " << node << std::endl;
    }
    return supported;
  }
  c10::optional<OpConverter> converter = GetMlirMhloConverter(node);
  try {
    if (converter) {
      mlir::DialectRegistry registry;
      RegisterDialects(registry);
      mlir::MLIRContext mlir_context(registry);
      mlir_context.loadAllAvailableDialects();
      MhloConversionContext empty_ctx(
          mlir_context, nullptr, /*is_support_testing*/ true);
      if (!node.kind().is_prim()) {
        if (!AllTensorTypeAnalyzed(node)) {
          return false;
        }
        // TODO: node that are not prim nodes, may require a subgraph
        //  as context to figure out whether it's supported
        if (SafeToCallConverter(node)) {
          return (*converter)(empty_ctx, node);
        }
        return true;
      } else {
        return (*converter)(empty_ctx, node);
      }
    }
  } catch (std::exception& err) {
    LOG(ERROR) << err.what();
    return false;
  }
  return false;
}

class ConvertToMhloImpl {
 public:
  ConvertToMhloImpl(
      std::shared_ptr<torch::jit::Graph> graph,
      mlir::MLIRContext& mlir_context)
      : cvt_context_(mlir_context, graph, /*is_support_testing*/ false) {}

  std::tuple<std::string, std::string, std::string, std::string> Run() {
    // make this function call only once.
    TORCH_CHECK(
        !converted_.test_and_set(std::memory_order_acquire),
        "the conversion is called multiple times");

    BuildMainFunc();
    RunImpl(cvt_context_.torch_graph->block());
    FinalizeMainFunc();

    std::string parsable_str =
        GenerateMlirModuleString(false, *cvt_context_.mlir_module);
    std::string pretty_str =
        GenerateMlirModuleString(true, *cvt_context_.mlir_module);
    return std::make_tuple(
        std::move(parsable_str),
        std::move(pretty_str),
        input_dev_str_,
        output_dev_str_);
  }

 private:
  void BuildMainFunc() {
    std::tie(mlir_main_func_, input_dev_str_, output_dev_str_) =
        CreateMlirFunction(
            cvt_context_,
            "main",
            cvt_context_.torch_graph->inputs(),
            cvt_context_.torch_graph->outputs());
    cvt_context_.mlir_module->push_back(mlir_main_func_);
  }

  void FinalizeMainFunc() {
    auto loc = cvt_context_.mlir_module->getLoc();
    SmallVec4<mlir::Value> return_values;
    auto& value_map = cvt_context_.value_map;
    for (auto output : cvt_context_.torch_graph->outputs()) {
      auto out_iter = value_map.find(output);
      TORCH_CHECK(
          out_iter != value_map.end(),
          "The mlir module output value was not found for %",
          output->debugName(),
          ". Please find the bug in the converter function for ",
          *output->node());
      return_values.push_back(out_iter->second);
    }
    cvt_context_.builder->create<mlir::func::ReturnOp>(loc, return_values);
    auto main_func_type = mlir_main_func_.getFunctionType();
    SmallVec4<mlir::Type> rets;
    for (auto& output : return_values) {
      rets.emplace_back(output.getType());
    }

    auto mlir_context = cvt_context_.mlir_module->getContext();
    auto new_mlir_func_type =
        mlir::FunctionType::get(mlir_context, main_func_type.getInputs(), rets);
    mlir_main_func_.setType(new_mlir_func_type);
  }

  void RunImpl(const torch::jit::Block* block) {
    for (auto node : block->nodes()) {
      for (auto inner_block : node->blocks()) {
        RunImpl(inner_block);
      }
      c10::optional<OpConverter> op_converter = GetMlirMhloConverter(*node);
      TORCH_CHECK(
          op_converter, *node, " hasn't been supported, please report a bug");
      // do conversion
      if (!((*op_converter)(cvt_context_, *node))) {
        cvt_context_.mlir_module->dump();
        TORCH_CHECK(false, "meet error during converting ", *node);
      }
    }
  }

  std::string input_dev_str_;
  std::string output_dev_str_;
  mlir::func::FuncOp mlir_main_func_;
  MhloConversionContext cvt_context_;
  std::atomic_flag converted_ = ATOMIC_FLAG_INIT;
};

std::tuple<std::string, std::string, std::string, std::string>
ConvertTorchToMhlo(std::shared_ptr<torch::jit::Graph> graph) {
  torch::blade::PropagateInputShapes(graph);
  GRAPH_DEBUG("TorchMhlo input graph:\n", *graph);
  std::shared_ptr<const torch::jit::Graph> const_graph = graph;
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  ::mlir::torch::registerAllDialects(registry);
  ::mlir::MLIRContext mlir_context(registry);
  mlir_context.loadAllAvailableDialects();
  ::llvm::SourceMgr sourceMgr;
  ::mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlir_context);

  mlir::OpPrintingFlags print_flags;
  print_flags.elideLargeElementsAttrs();

  bool enable_printing =
      env::ReadBoolFromEnvVar("TORCH_BLADE_MHLO_DEBUG_LOG", false);
  if (enable_printing) {
    mlir_context.disableMultithreading();
    mlir_context.printOpOnDiagnostic(true);
  }
  auto mlir_module =
      ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(&mlir_context));
  torch::jit::CompilationUnit unit;
  auto fn = unit.create_function("main", graph);
  auto op = torch_mlir::importJitFunctionAsFuncOp(wrap(&mlir_context), fn);
  auto builder = ::mlir::OpBuilder(&mlir_context);
  mlir_module.push_back(unwrap(op));

  ::mlir::torch::Torch::TorchLoweringPipelineOptions options;
  ::mlir::PassManager pm(
      &mlir_context, ::mlir::OpPassManager::Nesting::Implicit);
  if (enable_printing) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass*/ [](mlir::Pass*,
                                     mlir::Operation*) { return true; },
        /*shouldPrintAfterPasss*/
        [](mlir::Pass*, mlir::Operation*) { return true; },
        /*printModuleScope*/ false,
        /*printAfterOnlyOnChange*/ true,
        /*printAfterOnlyOnFailure*/ false,
        /*out*/ ::llvm::errs(),
        /*opPrintingFlags*/ print_flags);
  }
  ::mlir::torch::createDiscTorchBackendToMhloBackendPipeline(pm, options);
  if (mlir::failed(pm.run(mlir_module))) {
    mlir_module.emitError() << "TorchBackendToMhloBackendPipeline failed";
    return std::make_tuple("", "", "", "");
  }

  // Find the unique return op.
  ::mlir::func::FuncOp funcOp;
  ::mlir::WalkResult walkResult =
      mlir_module.walk([&](::mlir::func::FuncOp op) {
        if (funcOp)
          return ::mlir::WalkResult::interrupt();
        funcOp = op;
        return ::mlir::WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    mlir_module.emitError()
        << "unimplemented: refining returns for function with "
           "more than one return op";
    return std::make_tuple("", "", "", "");
  }
  auto entry_attr =
      GetFunctionAttrs(builder, const_graph->inputs(), const_graph->outputs());
  auto funcType = funcOp.getFunctionType();
  auto tf_func = builder.create<mlir::func::FuncOp>(
      funcOp.getLoc(),
      "main",
      funcType,
      ::llvm::ArrayRef<::mlir::NamedAttribute>{entry_attr});
  ::mlir::BlockAndValueMapping mapper;
  funcOp.cloneInto(tf_func, mapper);
  funcOp.erase();
  mlir_module.push_back(tf_func);

  std::string parsable_str = GenerateMlirModuleString(false, mlir_module);
  std::string pretty_str = GenerateMlirModuleString(true, mlir_module);
  return std::make_tuple(
      std::move(parsable_str),
      std::move(pretty_str),
      GetAttrString(GetDeviceStrs(const_graph->inputs())),
      GetAttrString(GetDeviceStrs(const_graph->outputs())));
}

std::tuple<std::string, std::string, std::string, std::string>
ConvertTorchScriptToMhlo(std::shared_ptr<torch::jit::Graph> graph) {
  try {
    if (IsTorchMlirAvailable()) {
      return ConvertTorchToMhlo(graph);
    } else {
      mlir::DialectRegistry registry;
      RegisterDialects(registry);
      mlir::MLIRContext mlir_context(registry);
      mlir_context.loadAllAvailableDialects();

      // will be deprecated soon.
      ConvertToMhloImpl impl(graph, mlir_context);
      return impl.Run();
    }
  } catch (std::exception& err) {
    LOG(ERROR) << err.what();
    return std::make_tuple("", "", "", "");
  }
}

bool IsTorchMlirAvailable() {
  return env::ReadBoolFromEnvVar("TORCH_DISC_USE_TORCH_MLIR", true);
}

} // namespace blade
} // namespace torch
