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

namespace mlir {
namespace torch {
namespace utils {
namespace env {
// These functions are copied from pytorch_blade/common_utils/,
// whose logger is replaced by the llvm's in order to
// resolve the the dependency problem.
std::string AsciiStrToLower(const char* cstr);
bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val);
std::string ReadStringFromEnvVar(
    const char* env_var_name,
    std::string default_val);
} // namespace env
} // namespace utils
} // namespace torch
} // namespace mlir