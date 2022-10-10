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

#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>
#include "pytorch_blade/common_utils/logging.h"

namespace torch {
namespace blade {

c10::OperatorName GetPrimOperatorName(const c10::Symbol& kind) {
  TORCH_CHECK(kind.is_prim());
  return c10::OperatorName(kind.toQualString(), "");
}

class MhloConverterRegistery {
 private:
  using ConverterLUT = std::unordered_map<c10::OperatorName, OpConverter>;
  MhloConverterRegistery() {}

 public:
  DISALLOW_COPY_AND_ASSIGN(MhloConverterRegistery);

  MhloConverterRegistery& RegisterPattern(
      const c10::OperatorName& op_name,
      const OpConverter& converter);
  MhloConverterRegistery& RegisterPattern(
      const torch::jit::FunctionSchema& signature,
      const OpConverter& converter);
  MhloConverterRegistery& RegisterPattern(
      const std::string& signature,
      const OpConverter& converter);

  c10::optional<OpConverter> GetNodeConverter(const torch::jit::Node& node);

  static MhloConverterRegistery& GetRegistery() {
    static MhloConverterRegistery registery;
    return registery;
  }

 private:
  ConverterLUT converter_lut_;
};

MhloConverterRegistery& MhloConverterRegistery::RegisterPattern(
    const c10::OperatorName& op_name,
    const OpConverter& converter) {
  converter_lut_[op_name] = converter;
  return *this;
}

MhloConverterRegistery& MhloConverterRegistery::RegisterPattern(
    const torch::jit::FunctionSchema& signature,
    const OpConverter& converter) {
  const auto& name = signature.operator_name();
  return RegisterPattern(name, converter);
}

MhloConverterRegistery& MhloConverterRegistery::RegisterPattern(
    const std::string& signature,
    const OpConverter& converter) {
  auto schema = torch::jit::parseSchema(signature);
  return RegisterPattern(schema, converter);
}

c10::optional<OpConverter> MhloConverterRegistery::GetNodeConverter(
    const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  ConverterLUT::iterator iter = converter_lut_.end();
  if (schema) {
    auto name = schema->operator_name();
    iter = converter_lut_.find(name);
  } else if (node.kind().is_prim()) {
    auto name = GetPrimOperatorName(node.kind());
    iter = converter_lut_.find(name);
  }
  if (iter != converter_lut_.end()) {
    return iter->second;
  }
  VLOG(2) << "Unable to get OpConverter for node: " << node;
  return c10::nullopt;
}

MhloConversionPatternRegister& MhloConversionPatternRegister::pattern(
    const std::string& schema,
    OpConverter converter) {
  auto& registery = MhloConverterRegistery::GetRegistery();
  registery.RegisterPattern(schema, converter);
  return *this;
}

MhloConversionPatternRegister& MhloConversionPatternRegister::pattern(
    const c10::OperatorName& op_name,
    OpConverter converter) {
  auto& registery = MhloConverterRegistery::GetRegistery();
  registery.RegisterPattern(op_name, converter);
  return *this;
}

c10::optional<OpConverter> GetMlirMhloConverter(const torch::jit::Node& node) {
  auto& registery = MhloConverterRegistery::GetRegistery();
  return registery.GetNodeConverter(node);
}

} // namespace blade
} // namespace torch
