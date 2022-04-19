// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/config.h>

#include "torch_disc/csrc/disc_compiler/disc_compiler.h"

namespace torch_disc {
namespace compiler {

struct CachedExecutable {
  explicit CachedExecutable(ExecutablePtr executable)
      : executable(std::move(executable)) {}

  ExecutablePtr executable;
};

using DiscComputationCache =
    torch::lazy::Cache<torch::lazy::hash_t, CachedExecutable,
                       torch::lazy::HashReducer>;

torch::lazy::BackendImplInterface* GetTSBackendImpl();

void InitTorchScriptBackend();

}  //  namespace compiler
}  //  namespace torch_disc
