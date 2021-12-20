#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {
namespace mhlo {
using mlir_dim_t = int64_t;
using mhlo_dim_t = int32_t;

template <class T>
using SmallVec4 = llvm::SmallVector<T, 4>;
using SmallValueVec4 = SmallVec4<mlir::Value>;

mlir::Type GetMlirTensorElemType(const mlir::Value& value);
mlir::RankedTensorType GetMilrRankedTensorType(const mlir::Value& tensor);
mlir::Type BuildMHloDimType(mlir::OpBuilder& builder);
mlir::Value TryBuildElementTypeCast(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& ml_tensor,
                                    const mlir::Type& elem_type);

// Build a fully dynamic mlir::RankedTensorType with the same rank and dtype
// to the input tensor
mlir::RankedTensorType BuildRankedTensorTypeFrom(const mlir::Value& tensor);
}  // namespace mhlo
}  // namespace mlir
