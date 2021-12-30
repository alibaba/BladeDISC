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

#include "compiler/mlir/converters/mhlo_conversion.h"
#include "compiler/mlir/pybind_functions.h"
#include "compiler/mlir/runtime/disc_engine_class.h"

#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace blade {

void initMLIRBindings(py::module& m) {
  py::module mlir =
      m.def_submodule("_mlir", "torch_blade python bindings to mlir");
  mlir.def("cvt_torchscript_to_mhlo", &ConvertTorchScriptToMhlo);
  mlir.def("is_mlir_mhlo_supported", &IsMlirMhloSupported);
  mlir.def("register_disc_engine", &register_disc_engine);
}

} // namespace blade
} // namespace torch
