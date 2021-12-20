#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildDotProduct_bmm(mlir::OpBuilder& builder,
                                const mlir::Location& loc,
                                const mlir::Value& inp_lhs,
                                const mlir::Value& inp_rhs);

mlir::Value BuildDotProduct_mm(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs);

mlir::Value BuildDotProduct_mv(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs);
}  // namespace mhlo
}  // namespace mlir