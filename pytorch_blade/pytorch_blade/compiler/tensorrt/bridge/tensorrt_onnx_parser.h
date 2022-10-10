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

#ifndef __TENSORRT_ONNX_PARSER_H__
#define __TENSORRT_ONNX_PARSER_H__

#include "NvOnnxParser.h"

#include "compiler/backends/engine_interface.h"
#include "compiler/tensorrt/bridge/tensorrt_calibrator.h"
#include "compiler/tensorrt/bridge/tensorrt_common.h"
#include "compiler/tensorrt/bridge/tensorrt_logger.h"

namespace torch {
namespace blade {

using torch::blade::backends::DynamicRanges;
using torch::blade::tensorrt::Int8EntropyCalibrator2;
struct OnnxParserContext {
  TrtUniquePtr<nvinfer1::IBuilder> builder;
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  TrtUniquePtr<nvonnxparser::IParser> parser;
  OnnxParserContext();
};

class TensorrtOnnxParser {
 public:
  TensorrtOnnxParser() {
    InitializeTrtPlugins(&GetTensorrtLogger());
  }
  bool IsSupportedModel(const std::string& proto_bytes);

  TrtUniquePtr<nvinfer1::ICudaEngine> BuildEngine(
      const std::string&,
      const std::shared_ptr<backends::EngineState>&,
      const std::vector<DynamicRanges>);

  void SetMaxWorkspaceSize(const size_t workspace_size) {
    max_workspace_size_ = workspace_size;
  }

 private:
  size_t max_workspace_size_ = 1 << 30; // 1GiB
};

bool platformHasFastInt8();

} // namespace blade
} // namespace torch
#endif // __TENSORRT_ONNX_PARSER_H__
