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

#include "pybind.h"

#include <mutex>
#include "common_utils/version.h"
#include "compiler/jit/onnx_funcs.h"
#include "compiler/jit/pybind_functions.h"
#ifdef TORCH_BLADE_BUILD_MLIR
#include "compiler/mlir/pybind_functions.h"
#endif // TORCH_BLADE_BUILD_MLIR)
#include "compiler/ltc/init_python_bindings.h"

// this include resolve some pybind11 incompatible problem of torch data
// structures like Dict
#include "torch/csrc/jit/python/pybind_utils.h"

namespace torch {
namespace blade {

namespace py = pybind11;
namespace {
std::string pybind_version() {
#define STR_EXPAND(token) #token
#define STR(token) STR_EXPAND(token)
  return "v" STR(PYBIND11_VERSION_MAJOR) "." STR(
      PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);
#undef STR
#undef STR_EXPAND
}
} // anonymous namespace

template <>
void initModules<COMMUNITY_VERSION_ID>(py::module& m) {
  torch::blade::initToolsBindings(m);
  m.def(
      "jit_pass_onnx_constant_f64_to_f32",
      &torch::blade::CastDownAllConstantDoubleToFloat);
#ifdef TORCH_BLADE_BUILD_MLIR
  torch::blade::initMLIRBindings(m);
#endif // TORCH_BLADE_BUILD_MLIR)
}

PYBIND11_MODULE(_torch_blade, m) {
  m.def("pybind_version", &pybind_version, R"pbdoc(
        return pybind version
    )pbdoc");
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  initModules<TORCH_BLADE_PLATFORM_VERSION_ID>(m);
  torch_disc::InitLtcModuleBindings(m);
}

} // namespace blade
} // namespace torch
