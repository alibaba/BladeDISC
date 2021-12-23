#include "compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>
#include "common_utils/logging.h"

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
  auto iter = converter_lut_.find(op_name);
  if (iter != converter_lut_.end()) {
    LOG(WARNING) << "Overriding already registered converter " << op_name
                 << ", unexpected behavior may occur";
  }
  converter_lut_[op_name] = converter;
  return *this;
}

MhloConverterRegistery& MhloConverterRegistery::RegisterPattern(
    const torch::jit::FunctionSchema& signature,
    const OpConverter& converter) {
  DLOG(INFO) << "Registering converter for " << signature;
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
  DLOG(INFO) << "Unable to get OpConverter for node: " << node;
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
