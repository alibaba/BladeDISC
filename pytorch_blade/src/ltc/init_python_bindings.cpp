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

#include "ltc/init_python_bindings.h"

#include "ltc/disc_backend/backend_impl.h"
#include "ltc/disc_compiler/replay.h"

namespace torch_disc {
namespace py = pybind11;
void InitLtcModuleBindings(py::module m) {
  py::module ltc = m.def_submodule(
      "_ltc", "torch_blade python bindings to PyTorch LazyTensor Core");

  ltc.def("_init_disc_backend", &compiler::InitTorchScriptBackend);
  ltc.def("load_and_replay", &compiler::LoadAndReplay);
}
} //  namespace torch_disc
