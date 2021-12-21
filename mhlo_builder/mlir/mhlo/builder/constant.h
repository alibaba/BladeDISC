#pragma once

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/macros.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
// Return true if the mlir::Value it produced from a mlir::mhlo::ConstOp
bool IsHloConstant(const mlir::Value&);
bool IsStdConstant(const mlir::Value&);
SmallVec4<int64_t> CastHloConstToListOfI64(const mlir::Value& value);

llvm::Optional<int64_t> CastStdConstToI64(const mlir::Value& val);
llvm::Optional<int64_t> CastHloConstToI64(const mlir::Value& val);
}  // namespace mhlo
}  // namespace mlir
