#pragma once

#include <string>
#include <vector>

#include "common_utils/macros.h"

namespace torch {
namespace addons {

std::vector<std::string> split(std::string line, const std::string& sep);
TorchAddonsDeclNewFlag(bool, TrustTracingShape);
TorchAddonsDeclNewFlag(bool, RecordClusterIOFlag);
} // namespace addons
} // namespace torch
