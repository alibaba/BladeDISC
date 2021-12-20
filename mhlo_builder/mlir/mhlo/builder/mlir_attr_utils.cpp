#include "mlir/mhlo/builder/mlir_attr_utils.h"

namespace mlir {
namespace mhlo {

// Returns 1D 64-bit dense elements attribute with the given values.
mlir::DenseIntElementsAttr BuildI64ElementsAttr(
    mlir::OpBuilder& builder, const mlir::ArrayRef<int64_t>& values) {
  mlir::RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, values);
}
}  // namespace mhlo
}  // namespace mlir