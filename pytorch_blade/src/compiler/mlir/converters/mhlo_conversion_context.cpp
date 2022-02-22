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

#include "compiler/mlir/converters/mhlo_conversion_context.h"

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/Dialect/StandardOps/IR/Ops.h> // from tf repo
#include "compiler/jit/tool_funcs.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

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

std::string GetAttrString(const SmallVec4<std::string>& str_vec) {
  std::string s;
  ::llvm::raw_string_ostream ss(s);
  ::llvm::interleave(str_vec, ss, ",");
  return ss.str();
}

std::tuple<mlir::FuncOp, std::string, std::string> CreateMlirFunction(
    MhloConversionContext& ctx,
    const std::string& function_name,
    at::ArrayRef<const torch::jit::Value*> inputs,
    at::ArrayRef<const torch::jit::Value*> outputs) {
  SmallVec4<mlir::Type> args;
  SmallVec4<mlir::Type> rets;
  SmallVec4<std::string> input_names;
  SmallVec4<std::string> output_names;
  SmallVec4<std::string> input_devices;
  SmallVec4<std::string> output_devices;

  auto& builder = *ctx.builder;
  for (auto& input : inputs) {
    auto mlir_tensor_type = BuildMlirRankedTensorType(builder, *input);
    args.emplace_back(mlir_tensor_type);
    input_names.push_back(input->debugName());
    // default to device cpu
    if (is_gpu_tensor_type(*input)) {
      input_devices.push_back("gpu");
    } else {
      input_devices.push_back("cpu");
    }
  }

  for (auto& output : outputs) {
    // The output type would be reset during the function building being
    // finalized. Currently it's set to placeholder unranked tensor type.
    auto unk_default_type = mlir::UnrankedTensorType::get(builder.getF32Type());
    rets.emplace_back(unk_default_type);
    output_names.push_back(output->debugName());
    // default to device cpu
    if (is_gpu_tensor_type(*output)) {
      output_devices.push_back("gpu");
    } else {
      output_devices.push_back("cpu");
    }
  }

  auto mlir_context = ctx.mlir_module->getContext();
  auto mlir_func_type = mlir::FunctionType::get(mlir_context, args, rets);
  SmallVec4<mlir::NamedAttribute> attrs;

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

  attrs.push_back(builder.getNamedAttr(
      "tf.entry_function",
      builder.getDictionaryAttr(
          {inputs_attr,
           outputs_attr,
           input_placements_attr,
           output_placements_attr})));

  ::llvm::ArrayRef<mlir::NamedAttribute> attr_arr = attrs;
  auto func = mlir::FuncOp::create(
      mlir::UnknownLoc::get(mlir_context),
      function_name,
      mlir_func_type,
      attr_arr);

  auto entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  size_t idx = 0;
  for (auto& input : inputs) {
    ctx.value_map[input] = func.getArgument(idx++);
  }

  return std::make_tuple(func, input_dev_str, output_dev_str);
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
