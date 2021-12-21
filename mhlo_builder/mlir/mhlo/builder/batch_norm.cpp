#include "mlir/mhlo/builder/batch_norm.h"

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildBatchNormInference(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const mlir::Value& scale,
    const mlir::Value& offset, const mlir::Value& mean,
    const mlir::Value& variance, float eps, mlir_dim_t feature_index) {
  mlir::FloatAttr eps_attr = builder.getF32FloatAttr(eps);
  mlir::IntegerAttr feature_index_attr =
      builder.getI64IntegerAttr(feature_index);

  auto bn_op = builder.create<mlir::mhlo::BatchNormInferenceOp>(
      loc, input.getType(), input, scale, offset, mean, variance, eps_attr,
      feature_index_attr);
  return bn_op.getResult();
}

}  // namespace mhlo
}  // namespace mlir
