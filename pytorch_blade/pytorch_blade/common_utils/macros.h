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

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete;

#define TorchBladeDeclNewFlag(T, Name) \
  T Set##Name(T);                      \
  T& Get##Name();

#define TorchBladeDefNewFlag(T, Name) \
  T& Get##Name() {                    \
    thread_local T flag;              \
    return flag;                      \
  }                                   \
  T Set##Name(T flag) {               \
    T old_flag = Get##Name();         \
    Get##Name() = flag;               \
    return old_flag;                  \
  }

#define TORCH_BLADE_RECORD_FUNCTION(func_name, inputs) \
  RECORD_FUNCTION(func_name, inputs)

#define PYTORCH_VERSION_GE(major, minor) \
  (PYTORCH_MAJOR_VERSION > major ||      \
   PYTORCH_MAJOR_VERSION == major && PYTORCH_MINOR_VERSION >= minor)

#define PYTORCH_VERSION_LE(major, minor) \
  (PYTORCH_MAJOR_VERSION < major ||      \
   PYTORCH_MAJOR_VERSION == major && PYTORCH_MINOR_VERSION <= minor)
