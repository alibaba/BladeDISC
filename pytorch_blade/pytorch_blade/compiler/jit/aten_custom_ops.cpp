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

#include <torch/script.h>
namespace torch {
namespace blade {
namespace {
#define WRAPPER_ADDSUB_TENSOR(func) \
  at::Tensor wrapper_##func(        \
      const at::Tensor& self,       \
      const at::Tensor& other,      \
      const at::Scalar& alpha) {    \
    return self.func(other, alpha); \
  }

#define WRAPPER_MULDIV_TENSOR(func)                                            \
  at::Tensor wrapper_##func(const at::Tensor& self, const at::Tensor& other) { \
    return self.func(other);                                                   \
  }

WRAPPER_ADDSUB_TENSOR(add);
WRAPPER_ADDSUB_TENSOR(add_);
WRAPPER_ADDSUB_TENSOR(sub);
WRAPPER_ADDSUB_TENSOR(sub_);
WRAPPER_MULDIV_TENSOR(mul);
WRAPPER_MULDIV_TENSOR(mul_);
WRAPPER_MULDIV_TENSOR(div);
WRAPPER_MULDIV_TENSOR(div_);

#undef WRAPPER_MULDIV_TENSOR
#undef WRAPPER_ADDSUB_TENSOR
} // namespace

namespace {
#define C10_REGISTER_OP(func, schema)                   \
  static auto reg_##func = c10::RegisterOperators().op( \
      schema,                                           \
      torch::RegisterOperators::options()               \
          .catchAllKernel(&wrapper_##func)              \
          .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))

C10_REGISTER_OP(
    add_,
    "aten::add_inplace_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
C10_REGISTER_OP(
    add,
    "aten::add_inplace.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");

C10_REGISTER_OP(
    sub_,
    "aten::sub_inplace_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
C10_REGISTER_OP(
    sub,
    "aten::sub_inplace.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");

C10_REGISTER_OP(
    mul_,
    "aten::mul_inplace_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
C10_REGISTER_OP(
    mul,
    "aten::mul_inplace.Tensor(Tensor self, Tensor other) -> Tensor");

C10_REGISTER_OP(
    div_,
    "aten::div_inplace_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
C10_REGISTER_OP(
    div,
    "aten::div_inplace.Tensor(Tensor self, Tensor other) -> Tensor");

#undef C10_REGISTER_OP
} // namespace

} // namespace blade
} // namespace torch
