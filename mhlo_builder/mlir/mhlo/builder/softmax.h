#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildSoftmax(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& ml_input, mlir_dim_t reduce_dim,
                         bool is_logsoftmax = false);
}  // namespace mhlo
}  // namespace mlir