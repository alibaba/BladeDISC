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

#include "compiler/mlir/converters/mhlo_conversion.h"

#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h>
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

#include "common_utils/logging.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

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
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::tensor::TensorDialect>(),
      registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  registry.insert<mlir::chlo::HloClientDialect>();
}

bool IsMlirMhloSupported(const torch::jit::Node& node) {
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
    DLOG(ERROR) << err.what();
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

  std::string GenerateMlirModuleString(bool pretty) {
    std::string s;
    ::llvm::raw_string_ostream ss(s);
    mlir::OpPrintingFlags print_flags;
    // IR generated with 'prettyForm' is not parsable.
    if (pretty) {
      print_flags.elideLargeElementsAttrs();
    }
    print_flags.enableDebugInfo(/*prettyForm*/ pretty);
    cvt_context_.mlir_module->print(ss, print_flags);
    return ss.str();
  }

  std::tuple<std::string, std::string, std::string, std::string> Run() {
    // make this function call only once.
    TORCH_CHECK(
        !converted_.test_and_set(std::memory_order_acquire),
        "the conversion is called multiple times");
    BuildMainFunc();
    RunImpl(cvt_context_.torch_graph->block());
    FinalizeMainFunc();

    std::string parsable_str = GenerateMlirModuleString(false);
    std::string pretty_str = GenerateMlirModuleString(true);
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
    cvt_context_.builder->create<mlir::ReturnOp>(loc, return_values);
    auto main_func_type = mlir_main_func_.getType();
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
        block->owningGraph()->dump();
        TORCH_CHECK(false, "meet error during converting ", *node);
      }
    }
  }

  std::string input_dev_str_;
  std::string output_dev_str_;
  mlir::FuncOp mlir_main_func_;
  MhloConversionContext cvt_context_;
  std::atomic_flag converted_ = ATOMIC_FLAG_INIT;
};

std::tuple<std::string, std::string, std::string, std::string>
ConvertTorchScriptToMhlo(std::shared_ptr<torch::jit::Graph> graph) {
  try {
    mlir::DialectRegistry registry;
    RegisterDialects(registry);
    mlir::MLIRContext mlir_context(registry);
    mlir_context.loadAllAvailableDialects();
    ConvertToMhloImpl impl(graph, mlir_context);
    return impl.Run();
  } catch (std::exception& err) {
    DLOG(ERROR) << err.what();
    return std::make_tuple("", "", "", "");
  }
}

} // namespace blade
} // namespace torch
