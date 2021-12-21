#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

// Build Convolution with the following operands layouts:
// input layout: N, IC, Spatial dims
// output layout: N, OC, Spatial dims
// kenerl layout: OC, IC, Spatial dims
mlir::Value BuildConvolution(mlir::OpBuilder& builder,
                             const mlir::Location& loc,
                             const mlir::Value& input,
                             const mlir::Value& weight,
                             const mlir::Value& padding,
                             mlir::ArrayRef<int64_t> strides,
                             mlir::ArrayRef<int64_t> dilations);

}  // namespace mhlo
}  // namespace mlir
