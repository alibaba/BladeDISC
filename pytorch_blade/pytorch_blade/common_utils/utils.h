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

#pragma once

#include <string>
#include <vector>

#include "pytorch_blade/common_utils/macros.h"

namespace c10 {
class IValue;
} // namespace c10

namespace torch {
namespace blade {

std::vector<std::string> split(std::string line, const std::string& sep);
TorchBladeDeclNewFlag(bool, TrustTracingShape);
TorchBladeDeclNewFlag(bool, RecordClusterIOFlag);

std::string AsciiStrToLower(const char* cstr);

// Dump IValues to file
void DumpIValues(
    const std::vector<c10::IValue>& inputs,
    const std::string& path);

std::vector<std::string> StrSplit(const std::string& str, char delim);

namespace env {
bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val);
double ReadDoubleFromEnvVar(const char* env_var_name, double default_val);
std::string ReadStringFromEnvVar(
    const char* env_var_name,
    std::string default_val);
} // namespace env
} // namespace blade
} // namespace torch
