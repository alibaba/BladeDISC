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

#include "tensorrt_common.h"

#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include "src/util/logging.h"
#include "tensorrt_logger.h"

namespace tf_blade {
namespace trt {
std::string GetLinkedTensorRTVersion() {
  std::stringstream ss;
  ss << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
     << NV_TENSORRT_PATCH;
  return ss.str();
}

std::string GetLoadedTensorRTVersion() {
  int ver = getInferLibVersion();
  int major = ver / 1000;
  ver = ver - major * 1000;
  int minor = ver / 100;
  int patch = ver - minor * 100;
  std::stringstream ss;
  ss << major << "." << minor << "." << patch;
  return ss.str();
}

// TODO: Unify error handling methods
std::string NvDataType2String(nvinfer1::DataType dtype) {
  static std::map<nvinfer1::DataType, const char*> d2s = {
      {nvinfer1::DataType::kFLOAT, "float32"},
      {nvinfer1::DataType::kHALF, "float16"},
      {nvinfer1::DataType::kINT8, "int8"},
      {nvinfer1::DataType::kINT32, "int32"},
      {nvinfer1::DataType::kBOOL, "bool"}};
  auto it = d2s.find(dtype);
  if (it != d2s.end()) {
    return it->second;
  } else {
    return std::string("");
  }
}

bool InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
  static std::once_flag flag;

  std::call_once(flag, [&]() {
    LOG(INFO) << "Linked TensorRT version: " << GetLinkedTensorRTVersion();
    LOG(INFO) << "Loaded TensorRT version: " << GetLoadedTensorRTVersion();
    if (!initLibNvInferPlugins(trt_logger, "")) {
      LOG(ERROR) << "Failed to initialize TensorRT plugins, and conversion may "
                    "fail later.";
    }
  });
  return true;
}

nvinfer1::ICudaEngine* CreateInferRuntime(const std::string& engine_data) {
  auto& logger = GetTensorrtLogger();
  // The initLibNvInferPlugins would be called once
  bool ret = InitializeTrtPlugins(&logger);
  if (ret) {
    const TrtUniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(logger)};
    auto* engine_ptr = runtime->deserializeCudaEngine(
        engine_data.data(), engine_data.size(), nullptr);
    return engine_ptr;
  } else {
    return nullptr;
  }
}

}  // namespace trt
}  // namespace tf_blade
