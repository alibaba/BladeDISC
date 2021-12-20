#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const llvm::Optional<mlir::Value>& var,
                              const llvm::Optional<mlir::Value>& mean,
                              double eps,
                              const SmallVec4<mlir_dim_t>& broadcast_dims);

mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input, double eps,
                              mlir_dim_t reduced_last_dims);

// math: y = input * gamma + beta
// where gamma and beta's dims must match last certain number dims of input
mlir::Value BuildElemAffine(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const llvm::Optional<mlir::Value>& gamma,
    const llvm::Optional<mlir::Value>& beta,
    const llvm::Optional<SmallVec4<mlir_dim_t>>& broadcast_dims);

}  // namespace mhlo
}  // namespace mlir
