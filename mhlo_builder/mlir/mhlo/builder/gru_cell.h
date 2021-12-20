#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildGRUCell(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& input, const mlir::Value& h_gates,
                         const mlir::Value& h_x, const mlir::Value& inp_bias,
                         const mlir::Value& h_bias);
}  // namespace mhlo
}  // namespace mlir