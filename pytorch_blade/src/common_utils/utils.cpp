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

#include <algorithm>
#include <cstdlib>
#include <string>

#include "common_utils/logging.h"

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

std::string AsciiStrToLower(const char* cstr) {
  if (cstr == nullptr) {
    return "";
  } else {
    std::string str(cstr);
    std::for_each(str.begin(), str.end(), [](char& c) { c = ::tolower(c); });
    return str;
  }
}

TorchBladeDefNewFlag(bool, TrustTracingShape);
TorchBladeDefNewFlag(bool, RecordClusterIOFlag);

namespace env {
bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val) {
  const char* env_var_val = std::getenv(env_var_name);
  if (env_var_val == nullptr) {
    return default_val;
  }

  std::string str_value = AsciiStrToLower(env_var_val);
  if (str_value == "0" || str_value == "false" || str_value == "off") {
    return false;
  } else if (str_value == "1" || str_value == "true" || str_value == "on") {
    return true;
  }
  LOG(ERROR) << "Failed to parse the env-var ${" << env_var_name
             << "} into bool: " << env_var_val
             << ". Use the default value: " << default_val;
  return default_val;
}

std::string ReadStringFromEnvVar(
    const char* env_var_name,
    std::string default_val) {
  const char* env_var_val = std::getenv(env_var_name);
  if (env_var_val == nullptr) {
    return default_val;
  }

  return std::string(env_var_val);
}
} // namespace env
} // namespace blade
} // namespace torch
