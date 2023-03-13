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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/disc/IR/hlo_disc_ops.h"

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/tool_funcs.h"
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"
#include "pytorch_blade/compiler/mlir/converters/torch_mlir_op_filter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"

#if PYTORCH_VERSION_LE(1, 8)
namespace c10 {
#undef LLVM_SUPPORT_MATHEXTRAS_H
#include <c10/util/llvmMathExtras.h>
#define LLVM_SUPPORT_MATHEXTRAS_H
} // namespace c10
#endif
#include "function_importer.h"

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
  print_flags.enableDebugInfo();

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
      &mlir_context,
      mlir_module.getOperationName(),
      ::mlir::OpPassManager::Nesting::Implicit);
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
  ::mlir::IRMapping mapper;
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
    return ConvertTorchToMhlo(graph);
  } catch (std::exception& err) {
    LOG(ERROR) << err.what();
    return std::make_tuple("", "", "", "");
  }
}
} // namespace blade
} // namespace torch
