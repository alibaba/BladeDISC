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

#include "pytorch_blade/common_utils/utils.h"

#include <torch/csrc/jit/serialization/pickle.h>
#include <algorithm>
#include <cstdlib>
#include <string>

#include "pytorch_blade/common_utils/logging.h"

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

void DumpIValue(const at::IValue& ivalue, const std::string& fname) {
  auto chars = torch::jit::pickle_save(ivalue);
  std::ofstream ofstream(fname, std::ios::out | std::ios::binary);
  ofstream.write(chars.data(), chars.size());
  ofstream.close();
}

void DumpIValues(
    const std::vector<c10::IValue>& inputs,
    const std::string& path) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto fname = path + "/" + std::to_string(i) + ".pt";
    DumpIValue(inputs[i], fname);
  }
}

std::vector<std::string> StrSplit(const std::string& str, char delim) {
  if (str.empty())
    return {};
  std::vector<std::string> ret;
  size_t first = 0;
  size_t next = str.find(delim);
  for (; next != std::string::npos;
       first = next + 1, next = str.find(delim, first)) {
    ret.push_back(str.substr(first, next - first));
  }
  ret.push_back(str.substr(first));
  return ret;
}

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

double ReadDoubleFromEnvVar(const char* env_var_name, double default_val) {
  double value = default_val;
  const char* env_var_val = std::getenv(env_var_name);
  if (env_var_val == nullptr) {
    return value;
  }
  try {
    value = std::strtod(env_var_val, nullptr);
    return value;
  } catch (std::runtime_error& error) {
    LOG(ERROR) << "Failed to parse the env-var ${" << env_var_name
               << "} into double: " << env_var_val
               << ". Use the default value: ",
        default_val;
  }
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
