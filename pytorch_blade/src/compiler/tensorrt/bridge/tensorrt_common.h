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

#ifndef __TENSORRT_PLUGINS_H__
#define __TENSORRT_PLUGINS_H__
#include <memory>

#include "NvInfer.h"
#include "NvInferPlugin.h"

namespace torch {
namespace blade {

std::string GetLinkedTensorRTVersion();
std::string GetLoadedTensorRTVersion();

std::string NvDataType2String(nvinfer1::DataType dtype);

void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger);
nvinfer1::ICudaEngine* CreateInferRuntime(const std::string& engine_data);

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
    if (t != nullptr) {
      t->destroy();
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

} // namespace blade
} // namespace torch
#endif //__TENSORRT_PLUGINS_H__
