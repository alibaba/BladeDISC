#include "mlir/mhlo/builder/constant.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
namespace mhlo {
bool IsHloConstant(const mlir::Value& value) {
  auto def = llvm::dyn_cast_or_null<mlir::mhlo::ConstOp>(value.getDefiningOp());
  return def != nullptr;
}

bool IsStdConstant(const mlir::Value& value) {
  auto def = llvm::dyn_cast_or_null<mlir::ConstantOp>(value.getDefiningOp());
  return def != nullptr;
}

template <typename T>
llvm::Optional<T> CastHloConstToElementsAttr(const mlir::Value& val) {
  auto def = llvm::dyn_cast<mlir::mhlo::ConstOp>(val.getDefiningOp());
  if (!def) {
    return llvm::None;
  }
  auto const_value = def.value();
  return const_value.dyn_cast_or_null<T>();
}

SmallVec4<int64_t> CastHloConstToListOfI64(const mlir::Value& value) {
  // cast will throw exception if meet error
  auto ml_elem_attr =
      CastHloConstToElementsAttr<mlir::DenseIntElementsAttr>(value);
  MHLO_CHECK(ml_elem_attr, "The input mlir::Value could not cast to const");
  SmallVec4<int64_t> vec_i64;
  // APInt: arbitrary precision integers.
  for (const auto& ap_index : ml_elem_attr->getValues<mlir::APInt>()) {
    int64_t index = ap_index.getSExtValue();
    vec_i64.push_back(index);
  }
  return vec_i64;
}

llvm::Optional<int64_t> CastAttrToI64(const mlir::Attribute& def) {
  auto attr = def.dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    int64_t index = attr.getValue().getSExtValue();
    return index;
  } else {
    return llvm::None;
  }
}

llvm::Optional<int64_t> CastHloConstToI64(const mlir::Value& val) {
  SmallVec4<int64_t> vec_i64 = CastHloConstToListOfI64(val);
  if (vec_i64.size() == 1) {
    return vec_i64[0];
  } else {
    return llvm::None;
  }
}

llvm::Optional<int64_t> CastStdConstToI64(const mlir::Value& val) {
  auto def = llvm::dyn_cast<mlir::ConstantOp>(val.getDefiningOp());
  if (!def) {
    return llvm::None;
  }
  return CastAttrToI64(def.value());
}
}  // namespace mhlo
}  // namespace mlir
