#pragma once

#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildGather(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& params, const mlir::Value& indices,
                        mlir_dim_t axis = 0);
}  // namespace mhlo
}  // namespace mlir
