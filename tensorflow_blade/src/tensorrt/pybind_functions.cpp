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

#include "pybind_functions.h"

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "bridge/tensorrt_flags.h"
#include "bridge/tensorrt_onnx_parser.h"
#include "src/util/logging.h"

#ifdef USE_TF_ALLOCATOR
#include "bridge/tensorrt_tf_allocator.h"
#include "cuda_runtime_api.h"
#include "src/util/tf_allocator_util.h"
#endif

namespace tf_blade {
namespace trt {
namespace {
using TMap = py::dict;

bool GetTensorRTBindingTypes(nvinfer1::ICudaEngine* engine, TMap& type_map) {
  CHECK_NOTNULL(engine);
  int num_bindings = engine->getNbBindings();
  for (int k = 0; k < num_bindings; ++k) {
    auto data_type = engine->getBindingDataType(k);
    const auto& name = engine->getBindingName(k);
    auto type = NvDataType2String(data_type);
    if (type.size() == 0) {
      return false;
    } else {
      type_map[name] = type;
    }
  }
  return true;
}

}  // namespace
bool is_onnx2trt_supported(const std::string& proto_bytes) {
  TensorrtOnnxParser parser;
  return parser.IsSupportedModel(proto_bytes);
}

py::tuple cvt_onnx_to_tensorrt(
    const std::string& dyn_proto_bytes,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const TrtDynamicRanges dynamic_ranges,
    const TensorrtOnnxParser::QValType& q_val) {
  TMap type_map;
  TensorrtOnnxParser parser;
  // The build_flags should not be changed
  auto build_flags = GetBuilderFlags();
#ifdef USE_TF_ALLOCATOR
  int cuda_device_id = 0;
  std::unique_ptr<TRTBaseAllocator> allocator;
  std::pair<int, Allocator*> device_allocator =
      tf_blade::util::GetDeviceAndAllocator();
  if (device_allocator.first >= 0) {
    cuda_device_id = device_allocator.first;
    allocator.reset(new TRTDeviceAllocator(device_allocator.second));
  }
  cudaSetDevice(cuda_device_id);
  auto engine = parser.BuildEngine(dyn_proto_bytes, input_shapes,
                                   dynamic_ranges, q_val, allocator.get());
#else
  // If value in allocator is a nullptr and cudamalloc will be used.
  auto engine = parser.BuildEngine(dyn_proto_bytes, input_shapes,
                                   dynamic_ranges, q_val, nullptr);
#endif

  if (engine != nullptr) {
    bool ret = GetTensorRTBindingTypes(engine.get(), type_map);
    if (ret) {
      auto out = engine->serialize();
      if (out != nullptr) {
        py::bytes engine_bytes((char*)out->data(), out->size());
        out->destroy();
        return py::make_tuple(engine_bytes, type_map, build_flags);
      }
    }
  }

  return py::make_tuple(py::bytes(""), type_map, build_flags);
}

// Some pybind functions specific for pytorch are removed,
// such as TrtEngineClass's register func
void initTensorRTBindings(py::module& m) {
  py::module trt = m.def_submodule("tensorrt", "python bindings to tensorrt");
  trt.def("is_onnx2trt_supported", &is_onnx2trt_supported);
  trt.def("cvt_onnx_to_tensorrt", &cvt_onnx_to_tensorrt);

  py::enum_<nvinfer1::BuilderFlag>(trt, "BuilderFlag", py::arithmetic())
      .value("FP16", nvinfer1::BuilderFlag::kFP16)
      .value("INT8", nvinfer1::BuilderFlag::kINT8)
      .value("DEBUG", nvinfer1::BuilderFlag::kDEBUG)
      .value("GPU_FALLBACK", nvinfer1::BuilderFlag::kGPU_FALLBACK)
      .value("STRICT_TYPES", nvinfer1::BuilderFlag::kSTRICT_TYPES)
      .value("REFIT", nvinfer1::BuilderFlag::kREFIT)
      .export_values();
  py::class_<TrtDynamicRanges>(trt, "TrtDynamicRanges")
      .def(py::init<const TrtDynamicRanges::single_shape&,
                    const TrtDynamicRanges::single_shape&,
                    const TrtDynamicRanges::single_shape&,
                    const std::vector<TrtDynamicRanges::single_shape>&>())
      .def(py::init<>());
  trt.def("set_builder_flags", &SetBuilderFlags,
          R"doc(
Setting TensorRT Engine Builder flags. Note this is per thread level
setting. For multithreading use cases, it needs to be configured
separately in each thread.
)doc");
  trt.def("get_builder_flags", &GetBuilderFlags,
          R"doc(
Getting TensorRT Engine Builder flags configured in current thread.
)doc");
  trt.def("_platform_has_fast_int8", &platformHasFastInt8);
}

}  // namespace trt
}  // namespace tf_blade
