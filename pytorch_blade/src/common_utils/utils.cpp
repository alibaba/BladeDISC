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
