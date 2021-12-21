
#include "compiler/mlir/converters/impl/utils.h"

#include "common_utils/logging.h"
#include "compiler/mlir/converters/impl/prim_constant.h"

#include <torch/script.h>

namespace torch {
namespace addons {
bool CheckConstAttribute(
    const torch::jit::Value* attr_val,
    const std::string& op_name,
    const std::string& param_name) {
  if (!IsPrimConstant(attr_val)) {
    TORCH_CHECK(attr_val != nullptr);
    DLOG(INFO) << "Could not convert " << op_name
               << " with non-compilation time parameter: " << param_name << " %"
               << attr_val->debugName();
    return false;
  }
  return true;
}
} // namespace addons
} // namespace torch
