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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_QUANTIZATION_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_QUANTIZATION_H_

// #if defined(TAO_CPU_ONLY)
#include <math.h>

#include <thread>

#include "mlir/ral/context/common_context_impl_mkldnn.h"
#include "mlir/ral/context/pdll_util.h"
#include "mlir/ral/device/cpu/cpu_driver.h"

namespace tao {
namespace ral {

inline void parseConvCustomeAttr(PDLAttr* attr, std::string& data_format_str,
                                 std::vector<int64_t>& dilations_arr,
                                 std::vector<int64_t>& strides_arr,
                                 std::string& padding_str,
                                 bool& weight_is_const,
                                 std::vector<int64_t>& explicit_paddings) {
  auto& dictAttr = attr->as<DictPDLAttr>();
  data_format_str = dictAttr.get("data_format").as<StrPDLAttr>().getValue();
  dilations_arr = dictAttr.get("dilations").as<IntArrayPDLAttr>().getValue();
  strides_arr = dictAttr.get("strides").as<IntArrayPDLAttr>().getValue();
  padding_str = dictAttr.get("padding").as<StrPDLAttr>().getValue();
  weight_is_const =
      dictAttr.get("weight_is_const").as<BoolPDLAttr>().getValue();
  explicit_paddings =
      dictAttr.get("explicit_paddings").as<IntArrayPDLAttr>().getValue();
}

template <int NDims>
inline bool generateReorderDims(std::string data_format_str,
                                std::vector<int32_t>& reorderDims) {
  int32_t batchRank = data_format_str.find("N");
  int32_t channelRank = data_format_str.find("C");
  if (batchRank == std::string::npos || channelRank == std::string::npos)
    return false;
  int idx = 0;
  for (int i = 0; i < NDims; ++i) {
    if (i == batchRank || i == channelRank) continue;
    reorderDims[idx++] = i;
  }
  reorderDims[idx++] = batchRank;
  reorderDims[idx++] = channelRank;
  return idx == NDims;
}

template <int NDims>
inline bool generateStrides(std::vector<int32_t> reorderDims,
                            std::vector<int64_t> strides_arr,
                            std::vector<int32_t>& strides) {
  if (strides_arr.size() == 1) {
    for (int i = 0; i < NDims - 2; ++i) strides[i] = strides_arr[0];
  } else if (strides_arr.size() == NDims - 2) {
    for (int i = 0; i < NDims - 2; ++i) strides[i] = strides_arr[i];
  } else if (strides_arr.size() == NDims) {
    for (int i = 0; i < NDims - 2; ++i)
      strides[i] = strides_arr[reorderDims[i]];
  } else {
    return false;
  }
  return true;
}

template <int NDims>
inline bool generateDilations(std::vector<int32_t> reorderDims,
                              std::vector<int64_t> dilations_arr,
                              std::vector<int32_t>& dilations) {
  if (dilations_arr.size() == 1) {
    for (int i = 0; i < NDims - 2; ++i) dilations[i] = dilations_arr[0];
  } else if (dilations_arr.size() == NDims - 2) {
    for (int i = 0; i < NDims - 2; ++i) dilations[i] = dilations_arr[i];
  } else if (dilations_arr.size() == NDims) {
    for (int i = 0; i < NDims - 2; ++i)
      dilations[i] = dilations_arr[reorderDims[i]];
  } else {
    return false;
  }
  return true;
}

// reference:
// https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
template <int NDims>
inline bool generatePadding(MemRefType<int8_t, NDims> input,
                            MemRefType<int8_t, NDims> weight,
                            std::string padding_str,
                            std::vector<int32_t> reorderDims,
                            std::vector<int32_t> strides,
                            std::vector<int32_t> dilations,
                            MemRefType<int32_t, 1>& padding) {
  if (padding_str == "SAME") {
    int padding_along = 0;
    int inputSize = 0, weightSize = 0, newWeightSize = 0;
    for (int i = 0; i < NDims - 2; ++i) {
      inputSize = input.sizes[reorderDims[i]];
      weightSize = weight.sizes[i + 1];
      newWeightSize = (weightSize - 1) * dilations[i] + 1;
      if (inputSize % strides[i] == 0)
        padding_along = std::max(int(newWeightSize - strides[i]), 0);
      else
        padding_along =
            std::max(int(newWeightSize - (inputSize % strides[i])), 0);
      padding.data[2 * i] = std::floor(padding_along / 2);
      padding.data[2 * i + 1] = padding_along - padding.data[2 * i];
    }
  } else if (padding_str == "VALID") {
    for (int i = 0; i < NDims - 2; ++i) {
      padding.data[2 * i] = 0;
      padding.data[2 * i + 1] = 0;
    }
  } else {
    return false;
  }
  return true;
}

template <int NDims>
inline bool generateExplicitPaddings(std::vector<int32_t> reorderDims,
                                     std::vector<int64_t> explicit_paddings,
                                     MemRefType<int32_t, 1>& padding) {
  int size = explicit_paddings.size();
  if (size == (NDims - 2) * 2) {
    for (int i = 0; i < NDims - 2; ++i) {
      padding.data[2 * i] = explicit_paddings[2 * i];
      padding.data[2 * i + 1] = explicit_paddings[2 * i + 1];
    }
  } else if (size == NDims * 2) {
    for (int i = 0; i < NDims - 2; ++i) {
      padding.data[2 * i] = explicit_paddings[2 * reorderDims[i]];
      padding.data[2 * i + 1] = explicit_paddings[2 * reorderDims[i] + 1];
    }
  } else {
    return false;
  }
  return true;
}

template <int NDims>
inline bool generateMetadata(std::vector<int32_t> strides,
                             std::vector<int32_t> dilations,
                             std::vector<int32_t> reorderDims,
                             bool weight_is_const,
                             MemRefType<int32_t, 1>& metadata) {
  int32_t batchRank = reorderDims[NDims - 2];
  int32_t channelRank = reorderDims[NDims - 1];
  int idx = 0;
  // input layout
  metadata.data[idx++] = batchRank;
  metadata.data[idx++] = channelRank;
  for (int i = 0; i < NDims - 2; ++i) metadata.data[idx++] = reorderDims[i];
  // kernel layout
  metadata.data[idx++] = NDims - 1;  // kernel input features
  metadata.data[idx++] = 0;          // kernel output features
  for (int i = 1; i < NDims - 1; ++i) metadata.data[idx++] = i;
  // output layout
  metadata.data[idx++] = batchRank;
  metadata.data[idx++] = channelRank;
  for (int i = 0; i < NDims - 2; ++i) metadata.data[idx++] = reorderDims[i];
  // strides
  for (auto stride : strides) metadata.data[idx++] = stride;
  // rhs_dilation
  for (auto dilation : dilations) metadata.data[idx++] = dilation;
  // is_weight_const
  metadata.data[idx++] = weight_is_const;

  return idx == metadata.sizes[0];
}

// refereces: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
template <int NDims>
inline void generateResultShape(MemRefType<int8_t, NDims> input,
                                MemRefType<int8_t, NDims> weight,
                                std::vector<int32_t> reorderDims,
                                MemRefType<int32_t, 1> padding,
                                std::vector<int32_t> strides,
                                std::vector<int32_t> dilations,
                                std::vector<int64_t>& resultSizes) {
  for (int i = 0; i < NDims - 2; ++i) {
    int rank = reorderDims[i];
    int ori = input.sizes[rank];   // input spatial dims
    int k = weight.sizes[i + 1];   // kernel spatial dims
    int s = strides[i];            // stride
    int d = dilations[i];          // dilation
    int p1 = padding.data[2 * i];  // padding
    int p2 = padding.data[2 * i + 1];
    resultSizes[rank] = (ori + p1 + p2 - k * d + d + s - 1) / s;
  }
  int batchRank = reorderDims[NDims - 2];
  int channelRank = reorderDims[NDims - 1];
  resultSizes[batchRank] = input.sizes[batchRank];
  resultSizes[channelRank] = weight.sizes[0];
}

}  // namespace ral
}  // namespace tao

// #endif  // defined(TAO_CPU_ONLY)

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_QUANTIZATION_H_
