#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildSigmoid(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& mlir_val);

}  // namespace mhlo
}  // namespace mlir