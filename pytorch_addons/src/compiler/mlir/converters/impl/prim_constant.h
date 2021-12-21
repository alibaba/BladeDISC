#pragma once
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/mhlo/builder/constant.h>

namespace at {
class Tensor;
}
namespace torch {
namespace jit {
class Value;
}
} // namespace torch

namespace torch {
namespace addons {

// Return true if the torch::jit::Value it produced from an prim::Constant
bool IsPrimConstant(const torch::jit::Value& val);
// Return true if the torch::jit::Value it produced from an prim::Constant
bool IsPrimConstant(const torch::jit::Value* val);

// Cast a prim::Constant of scalar list to type T
template <typename T>
std::vector<T> CastJitConstListToVec(const torch::jit::Value& val);

// Cast a prim::Constant of numeric scalar value to int64_t
int64_t CastJitConstToInt64(const torch::jit::Value& val);
// Cast a prim::Constant of numeric scalar value to double
double CastJitConstToDouble(const torch::jit::Value& val);
// Cast a prim::Constant of numeric scalar value to bool
bool CastJitConstToBool(const torch::jit::Value& val);

mlir::Value BuildMlirConstFromTorchTensor(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const at::Tensor& val);
} // namespace addons
} // namespace torch
