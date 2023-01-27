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

#include "decoupling/compiler_base.h"

namespace tensorflow {
namespace tao {

using Factories =
    std::unordered_map<std::string, CompilerBase::CompilerFactory>;
using Compilers =
    std::unordered_map<std::string, std::unique_ptr<CompilerBase>>;

static Factories* GetCompilerFactories() {
  static Factories factories;
  return &factories;
}

static Compilers* GetCompilers() {
  static Compilers compilers;
  return &compilers;
}

/* static */ void CompilerBase::RegisterCompilerFactory(
    DeviceType device_type, CompilerFactory factory) {
  auto* factories = GetCompilerFactories();
  std::string dt_string = device_type.type_string();
  CHECK(factories->find(dt_string) == factories->end())
      << "Compiler factory already registered for device " << dt_string;
  (*factories)[dt_string] = std::move(factory);
}

/* static */ StatusOr<CompilerBase*> CompilerBase::GetCompilerForDevice(
    DeviceType device_type) {
  std::string dt_string = device_type.type_string();
  auto* compilers = GetCompilers();

  {
    auto it = compilers->find(dt_string);
    if (it != compilers->end()) {
      return it->second.get();
    }
  }

  auto* factories = GetCompilerFactories();
  auto it = factories->find(dt_string);
  if (it == factories->end()) {
    return xla::NotFound(
        "could not find registered compiler wrapper for device %s -- check "
        "target linkage",
        dt_string);
  }

  compilers->insert(std::make_pair(dt_string, it->second()));
  return compilers->at(dt_string).get();
}

}  //  namespace tao
}  //  namespace tensorflow