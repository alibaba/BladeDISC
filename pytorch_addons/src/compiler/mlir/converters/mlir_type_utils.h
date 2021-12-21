#pragma once
#include <c10/core/ScalarType.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace torch {
namespace jit {
class Value;
}
} // namespace torch

namespace torch {
namespace addons {

mlir::Type BuildMlirElemType(
    mlir::Builder& builder,
    c10::ScalarType scalar_type);

mlir::RankedTensorType BuildMlirRankedTensorType(
    mlir::OpBuilder& builder,
    const torch::jit::Value& value,
    bool static_shape = false);

::llvm::Optional<mlir::Value> BuildCastWithJitType(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const mlir::Value& value,
    const torch::jit::Value* dtype);
} // namespace addons
} // namespace torch
