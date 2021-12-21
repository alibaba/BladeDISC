#pragma once

#include <string>

namespace torch {
namespace jit {
class Value;
}
} // namespace torch

namespace torch {
namespace addons {
bool CheckConstAttribute(
    const torch::jit::Value* attr_val,
    const std::string& op_name,
    const std::string& param_name);
} // namespace addons
} // namespace torch
