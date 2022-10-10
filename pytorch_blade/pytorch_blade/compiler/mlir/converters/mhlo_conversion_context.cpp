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

#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

MhloConversionContext::MhloConversionContext(
    mlir::MLIRContext& context,
    std::shared_ptr<torch::jit::Graph> graph,
    bool is_support_testing)
    : torch_graph(graph), is_support_testing_(is_support_testing) {
  mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  builder = std::make_shared<mlir::OpBuilder>(&context);
}

bool MhloConversionContext::IsSupportTesting() {
  return is_support_testing_;
}

mlir::Value MhloConversionContext::GetMlirValue(const torch::jit::Value* val) {
  auto found = value_map.find(val);
  TORCH_CHECK(
      found != value_map.end(),
      "mlir::Value for %",
      val->debugName(),
      " not found, please report a bug");
  return found->second;
}

::llvm::Optional<mlir::Value> MhloConversionContext::GetOptionalMlirValue(
    const torch::jit::Value* val) {
  auto found = value_map.find(val);
  if (found != value_map.end()) {
    // the value was found, it means the weight exists.
    return found->second;
  }

  // If optional weights is not found in value_map, it must be None.
  auto jit_ival = torch::jit::toIValue(val);
  TORCH_CHECK(val == nullptr || val->mustBeNone());
  return ::llvm::None;
}

const SmallVec4<mlir::Value>& MhloConversionContext::GetMlirValueList(
    const torch::jit::Value* val) {
  auto found = list_map.find(val);
  TORCH_CHECK(
      found != list_map.end(),
      "mlir::Value for %",
      val->debugName(),
      " not found, please report a bug");
  return found->second;
}

bool MhloConversionContext::IsSameContext(mlir::Value val) {
  return val.getContext() == mlir_module->getContext();
}

mlir::Location GetNodeLocation(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& source_range = node.sourceRange();
  if (auto file_line_col = source_range.file_line_col()) {
    std::string filename;
    size_t line, col;
    std::tie(filename, line, col) = *file_line_col;
    std::string node_kind_name(node.kind().toDisplayString());
    return mlir::FileLineColLoc::get(
        ctx.builder->getStringAttr(node_kind_name + "@" + filename), line, col);
  } else {
    // TODO: if unkown, return the block's location
    return ctx.mlir_module->getLoc();
  }
}

} // namespace blade
} // namespace torch
