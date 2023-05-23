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

#include "pytorch_blade/pybind.h"

#include <mutex>
#include "compiler/backends/engine_class.h"
#include "compiler/backends/engine_interface.h"
#include "compiler/jit/onnx_funcs.h"
#include "compiler/jit/pybind_functions.h"
#include "compiler/jit/torch/constant_propagation.h"
#include "compiler/jit/torch/shape_analysis.h"

#ifdef TORCH_BLADE_BUILD_QUANTIZATION
#include "quantization/pybind_functions.h"
#endif // TORCH_BLADE_BUILD_QUANTIZATION

#ifdef TORCH_BLADE_BUILD_TENSORRT
#include "compiler/tensorrt/pybind_functions.h"
#endif // TORCH_BLADE_BUILD_TENSORRT

#ifdef TORCH_BLADE_BUILD_NEURAL_ENGINE
#include "compiler/neural_engine/pybind_functions.h"
#endif

#ifdef TORCH_BLADE_BUILD_MLIR
#include "compiler/mlir/pybind_functions.h"
#endif // TORCH_BLADE_BUILD_MLIR

#ifdef TORCH_BLADE_ENABLE_LTC
#include "ltc/init_python_bindings.h"
#endif // TORCH_BLADE_ENABLE_LTC

// this include resolve some pybind11 incompatible problem of torch data
// structures like Dict
#include "torch/csrc/jit/python/pybind_utils.h"

namespace torch {
namespace blade {

constexpr bool is_platform_alibaba = TORCH_BLADE_PLATFORM_ALIBABA;

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
void initModules<false>(py::module& m) {
  torch::blade::initToolsBindings(m);
  m.def(
       "jit_pass_onnx_constant_f64_to_f32",
       &torch::blade::CastDownAllConstantDoubleToFloat)
      .def(
          "jit_pass_propagate_input_shapes",
          &torch::blade::PropagateInputShapes)
      .def("jit_pass_constant_propagation", &torch::blade::ConstantPropagation);
  m.def("hash_data", [&](const std::string& value) -> size_t {
    return c10::hash<std::string>{}(value);
  });
  m.def("hash_combine", [&](size_t a, size_t b) -> size_t {
    return c10::hash_combine(a, b);
  });

#ifdef TORCH_BLADE_BUILD_QUANTIZATION
  torch::blade::quantization::initQuantizationBindings(m);
#endif // TORCH_BLADE_BUILD_QUANTIZATION

#ifdef TORCH_BLADE_BUILD_TENSORRT
  torch::blade::tensorrt::initTensorRTBindings(m);
#endif // TORCH_BLADE_BUILD_TENSORRT

#ifdef TORCH_BLADE_BUILD_NEURAL_ENGINE
  torch::blade::neural_engine::initNeuralEngineBindings(m);
#endif // TORCH_BLADE_BUILD_NEURAL_ENGINE

#ifdef TORCH_BLADE_BUILD_MLIR
  torch::blade::disc::initMLIRBindings(m);
#endif // TORCH_BLADE_BUILD_MLIR

#ifdef TORCH_BLADE_ENABLE_LTC
  torch_disc::InitLtcModuleBindings(m);
#endif // TORCH_BLADE_ENABLE_LTC

  using namespace torch::blade::backends;
  py::module backends =
      m.def_submodule("_backends", "torch_blade python bindings to backends");

  py::class_<DynamicRanges>(backends, "DynamicRanges")
      .def(py::init<>())
      .def_readwrite("min_shape", &DynamicRanges::min_shape)
      .def_readwrite("max_shape", &DynamicRanges::max_shape)
      .def_readwrite("dynamic_setting", &DynamicRanges::dynamic_setting)
      .def_readwrite("opt_shapes", &DynamicRanges::opt_shapes)
      .def("validate", &DynamicRanges::Validate);

  py::class_<TensorInfo>(backends, "TensorInfo")
      .def(py::init<>())
      .def(py::init<const torch::jit::Value&>())
      .def_readwrite("sizes", &TensorInfo::sizes)
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("device", &TensorInfo::device)
      .def_property("dtype", &TensorInfo::GetDType, &TensorInfo::SetDType);

  py::class_<EngineState, std::shared_ptr<EngineState>>(backends, "EngineState")
      .def(py::init<>())
      .def_readwrite("engine_bytes", &EngineState::engine_bytes)
      .def_readwrite("model_proto", &EngineState::model_proto)
      .def_readwrite("backend_name", &EngineState::backend_name)
      .def_readwrite("inputs", &EngineState::inputs)
      .def_readwrite("outputs", &EngineState::outputs)
      .def_readwrite("calib_data", &EngineState::calib_data)
      .def_readwrite("extra_attrs", &EngineState::extra_attrs);

  backends.def("register_engine", register_engine);
  backends.def("create_engine", [](const EngineState& state) {
    auto serialized = EngineClass::Serialize(
        state, EngineInterface::GetBackendName(), "", "");
    return torch::jit::Object(
        torch::make_custom_class<EngineClass>(std::move(serialized))
            .toObject());
  });
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

  initModules<is_platform_alibaba>(m);
}

} // namespace blade
} // namespace torch
