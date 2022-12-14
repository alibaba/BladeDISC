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

#include "pytorch_blade/compiler/neural_engine/neural_engine.h"
#include "pytorch_blade/pybind.h"

#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace blade {
namespace neural_engine {
void initNeuralEngineBindings(py::module& m) {
  py::module neural_engine = m.def_submodule(
      "_neural_engine", "torch_blade python bindings to neural_engine");
  neural_engine.def("backend_name", &GetBackendName);
}
} // namespace neural_engine
} // namespace blade
} // namespace torch