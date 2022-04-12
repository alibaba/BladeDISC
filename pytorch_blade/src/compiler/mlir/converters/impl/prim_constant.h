// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
namespace blade {

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
// Cast a prim::Constant of numeric scalar value to string
std::string CastJitConstToString(const torch::jit::Value& val);

mlir::Value BuildMlirConstFromTorchTensor(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const at::Tensor& val);
} // namespace blade
} // namespace torch
