#pragma once

#include <string>
#include <vector>

#include "common_utils/macros.h"

namespace torch {
namespace blade {

std::vector<std::string> split(std::string line, const std::string& sep);
TorchBladeDeclNewFlag(bool, TrustTracingShape);
TorchBladeDeclNewFlag(bool, RecordClusterIOFlag);
} // namespace blade
} // namespace torch
