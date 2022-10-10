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

#pragma once
#include <llvm/ADT/Optional.h>
#include <unordered_map>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/mhlo/builder/mlir_type_utils.h>

#include <c10/util/ArrayRef.h>

#include "pytorch_blade/common_utils/macros.h"

namespace torch {
namespace jit {
class Graph;
class Node;
class Value;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {

struct MhloConversionContext {
  MhloConversionContext(const MhloConversionContext&) = delete;
  void operator=(const MhloConversionContext&) = delete;

  // The converter has 3 stages:
  // stage 1: per-node(torch::jit::Node) support information query
  // stage 2: supported nodes(torch::jit::Node) clustering as subgraph
  // stage 3: subgraph(torch::jit::Graph) to mlir::Module conversion
  //
  // To make the mechanism more reusable, cohesive, and less error-prone,
  // we provide a design that use the exactly same converter function for both
  // support information query and Op conversion.
  //
  // When is_support_testing flag is enable in MhloConversionContext,
  // the converter function should do suppport information query.
  MhloConversionContext(
      mlir::MLIRContext& context,
      std::shared_ptr<torch::jit::Graph> graph,
      bool is_support_testing);

  mlir::Value GetMlirValue(const torch::jit::Value* val);
  ::llvm::Optional<mlir::Value> GetOptionalMlirValue(
      const torch::jit::Value* val);

  const mlir::mhlo::SmallVec4<mlir::Value>& GetMlirValueList(
      const torch::jit::Value* val);
  bool IsSameContext(mlir::Value);
  bool IsSupportTesting();

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  std::shared_ptr<mlir::OpBuilder> builder;
  std::shared_ptr<const torch::jit::Graph> torch_graph;
  std::unordered_map<const torch::jit::Value*, mlir::Value> value_map;

  // All Converter-Time values of list type would be stored in the list_map,
  // and would be unpack into MLIR values after conversions.
  // In other words, there is no list types after conversion.
  std::unordered_map<
      const torch::jit::Value*,
      mlir::mhlo::SmallVec4<mlir::Value>>
      list_map;

 private:
  bool is_support_testing_;
};

mlir::Location GetNodeLocation(
    MhloConversionContext& ctx,
    const torch::jit::Node& node);

} // namespace blade
} // namespace torch
