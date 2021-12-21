#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildBroadcastScalarAsTensor(mlir::OpBuilder& builder,
                                         const mlir::Location& loc,
                                         const mlir::Value& scalar,
                                         const mlir::Value& tensor);

mlir::Value BuildBroadcastTensorAsOther(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const mlir::Value& other);

// Broadcast Tensor to shape with dims_size
mlir::Value BuildBroadcastTensorInDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& dims_size,
    const SmallVec4<mlir_dim_t>& broadcast_dims);
}  // namespace mhlo
}  // namespace mlir