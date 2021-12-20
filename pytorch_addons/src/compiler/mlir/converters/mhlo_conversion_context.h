#pragma once
#include <llvm/ADT/Optional.h>
#include <unordered_map>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/mhlo/builder/mlir_type_utils.h>

#include <c10/util/ArrayRef.h>

#include "common_utils/macros.h"

namespace torch {
namespace jit {
class Graph;
class Node;
class Value;
} // namespace jit
} // namespace torch

namespace torch {
namespace addons {

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

  mlir::OwningModuleRef mlir_module;
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

std::tuple<mlir::FuncOp, std::string, std::string> CreateMlirFunction(
    MhloConversionContext& ctx,
    const std::string& function_name,
    at::ArrayRef<const torch::jit::Value*> inputs,
    at::ArrayRef<const torch::jit::Value*> outputs);

mlir::Location GetNodeLocation(
    MhloConversionContext& ctx,
    const torch::jit::Node& node);

} // namespace addons
} // namespace torch
