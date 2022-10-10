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

#include "pytorch_blade/compiler/tensorrt/pybind_functions.h"

#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_flags.h"
#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_onnx_parser.h"
#include "pytorch_blade/compiler/tensorrt/tensorrt_engine.h"
#include "torch/csrc/jit/python/pybind_utils.h"

namespace torch {
namespace blade {
namespace tensorrt {

bool is_onnx2trt_supported(const std::string& proto_bytes) {
  TensorrtOnnxParser parser;
  return parser.IsSupportedModel(proto_bytes);
}

std::shared_ptr<backends::EngineState> cvt_onnx_to_tensorrt(
    const std::string& dyn_proto_bytes,
    const std::shared_ptr<backends::EngineState>& state,
    const std::vector<DynamicRanges> dynamic_ranges) {
  TensorrtOnnxParser parser;
  auto engine = parser.BuildEngine(dyn_proto_bytes, state, dynamic_ranges);
  if (engine != nullptr) {
    auto out = engine->serialize();
    if (out != nullptr) {
      state->engine_bytes =
          std::move(std::string((char*)out->data(), out->size()));
      state->backend_name = GetBackendName();
      out->destroy();
    }
    return state;
  } else {
    return nullptr;
  }
}

void initTensorRTBindings(py::module& m) {
  py::module trt =
      m.def_submodule("_tensorrt", "torch_blade python bindings to tensorrt");
  trt.def("is_onnx2trt_supported", &is_onnx2trt_supported);
  trt.def("cvt_onnx_to_tensorrt", &cvt_onnx_to_tensorrt);
  trt.def("backend_name", &GetBackendName);

  py::enum_<nvinfer1::BuilderFlag>(
      trt, "BuilderFlag", py::module_local(), py::arithmetic())
      .value("FP16", nvinfer1::BuilderFlag::kFP16)
      .value("INT8", nvinfer1::BuilderFlag::kINT8)
      .value("DEBUG", nvinfer1::BuilderFlag::kDEBUG)
      .value("GPU_FALLBACK", nvinfer1::BuilderFlag::kGPU_FALLBACK)
      .value("STRICT_TYPES", nvinfer1::BuilderFlag::kSTRICT_TYPES)
      .value("REFIT", nvinfer1::BuilderFlag::kREFIT)
      .export_values();

  trt.def(
      "set_builder_flags",
      &SetBuilderFlags,
      R"doc(
Setting TensorRT Engine Builder flags. Note this is per thread level
setting. For multithreading use cases, it needs to be configured
separately in each thread.
)doc");
  trt.def(
      "get_builder_flags",
      &GetBuilderFlags,
      R"doc(
Getting TensorRT Engine Builder flags configured in current thread.
)doc");
  trt.def("platform_has_fast_int8", &platformHasFastInt8);
}

} // namespace tensorrt
} // namespace blade
} // namespace torch
