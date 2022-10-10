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

#include <ATen/core/operator_name.h>
#include <c10/util/Optional.h>

#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"

namespace c10 {
class Symbol;
} // namespace c10
namespace torch {
namespace jit {
class Node;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {
c10::OperatorName GetPrimOperatorName(const c10::Symbol& kind);

typedef std::function<bool(MhloConversionContext&, const torch::jit::Node&)>
    OpConverter;
struct ConversionPattern {
  std::string schema;
  OpConverter converter;
};

struct MhloConversionPatternRegister {
  MhloConversionPatternRegister& pattern(
      const std::string& schema,
      OpConverter);
  MhloConversionPatternRegister& pattern(const c10::OperatorName&, OpConverter);
};

c10::optional<OpConverter> GetMlirMhloConverter(const torch::jit::Node&);
} // namespace blade
} // namespace torch
