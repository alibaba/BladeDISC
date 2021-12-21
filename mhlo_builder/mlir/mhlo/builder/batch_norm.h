#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildBatchNormInference(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const mlir::Value& scale,
    const mlir::Value& offset, const mlir::Value& mean,
    const mlir::Value& variance, float eps, mlir_dim_t feature_index);

}  // namespace mhlo
}  // namespace mlir
