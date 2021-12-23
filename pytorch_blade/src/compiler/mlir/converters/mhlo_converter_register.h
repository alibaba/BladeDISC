#pragma once

#include <ATen/core/operator_name.h>
#include <c10/util/Optional.h>

#include "common_utils/macros.h"
#include "compiler/mlir/converters/mhlo_conversion_context.h"

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
