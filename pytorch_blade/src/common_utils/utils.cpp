#include "common_utils/utils.h"

namespace torch {
namespace blade {
std::vector<std::string> split(std::string line, const std::string& sep) {
  std::vector<std::string> result;
  // Previously, we implemented split with std::sregex_token_iterator.
  // But we meets segfault when linked against torch, witch was compiled with
  // lower gcc version.
  //
  // So, We changed to a more naive implementation that works.
  size_t pos = 0;
  size_t offset = 0;
  while ((pos = line.find(sep, offset)) != std::string::npos) {
    auto token = line.substr(offset, pos - offset);
    result.emplace_back(token);
    offset = pos + sep.length();
  }
  if (offset < line.length()) {
    result.emplace_back(line.substr(offset));
  }
  return result;
}

TorchBladeDefNewFlag(bool, TrustTracingShape);
TorchBladeDefNewFlag(bool, RecordClusterIOFlag);
} // namespace blade
} // namespace torch
