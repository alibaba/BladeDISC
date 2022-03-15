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

#include <unordered_map>

#include "NvOnnxParser.h"
#include "tensorrt_common.h"
#include "tensorrt_logger.h"

namespace tf_blade {
namespace trt {
struct OnnxParserContext {
  TrtUniquePtr<nvinfer1::IBuilder> builder;
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  TrtUniquePtr<nvonnxparser::IParser> parser;
  OnnxParserContext();
};

struct TrtDynamicRanges {
  using single_shape = std::vector<std::vector<int64_t>>;
  TrtDynamicRanges(const single_shape& min_shape, const single_shape& max_shape,
                   const single_shape& dynamic_setting,
                   const std::vector<single_shape>& opt_shapes)
      : min_shape(min_shape),
        max_shape(max_shape),
        dynamic_setting(dynamic_setting),
        opt_shapes(opt_shapes) {}
  TrtDynamicRanges()
      : min_shape(), max_shape(), dynamic_setting(), opt_shapes() {}
  single_shape min_shape;
  single_shape max_shape;
  // dynamic dimension should be set to -1 when parse onnx to trt network.
  // we store this information in dynamic_setting.
  single_shape dynamic_setting;
  std::vector<single_shape> opt_shapes;
};

class TensorrtOnnxParser {
 public:
  using QValType = std::unordered_map<std::string, std::pair<float, float>>;

  TensorrtOnnxParser() { InitializeTrtPlugins(&GetTensorrtLogger()); }
  bool IsSupportedModel(const std::string& proto_bytes);

  TrtUniquePtr<nvinfer1::ICudaEngine> BuildEngine(
      const std::string&, const std::vector<std::vector<int64_t>>&,
      const TrtDynamicRanges dynamic_ranges, const QValType&,
      nvinfer1::IGpuAllocator* allocator = nullptr);

  void SetMaxWorkspaceSize(const size_t workspace_size) {
    max_workspace_size_ = workspace_size;
  }
  bool setDynamicRange(TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
                       const QValType& q_val);

 private:
  size_t max_workspace_size_ = 1 << 30;  // 1GiB
};

bool platformHasFastInt8();

}  // namespace trt
}  // namespace tf_blade
#endif  // __TENSORRT_ONNX_PARSER_H__
