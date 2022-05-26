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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_MKLDNN_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_MKLDNN_H_

#if defined(TAO_CPU_ONLY) && defined(TAO_ENABLE_MKLDNN)

#include "dnnl_threadpool_iface.hpp"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/mkldnn/ideep/ideep.hpp"
#include "tensorflow/compiler/mlir/xla/ral/device/cpu/cpu_driver.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_base.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"

namespace tao {
namespace ral {

enum DiscCpuMathKernelMode {
  kDiscPreferOneDNN = 1,
  kDiscPreferMKL = 0,
  kDiscPreferTuningBasedSelection = 2
};

// Returns the current active math kernel mode.
DiscCpuMathKernelMode GetDiscCpuMathKernelMode();

// Returns true if we prefer to promote all conv1d kernel to conv2d
bool promoteConv1DToConv2D();

// Returns true if weight prepacking is enabled.
bool enableWeightPrePacking();

// Returns the maximum number of copied we can cache.
int getWeightPrePackingCacheCapacity();

using ideep::data_type;
using ideep::dims;
using ideep::format_tag;
using ideep::tensor;

// Convert string format to format_tag
// Returns format_tag::undef if failed or not supported.
format_tag str2format(const std::string& fmt);

// convert c++ data type to onednn data type.
template <typename T>
data_type toDataType() {
  return data_type::undef;
}

template <>
inline data_type toDataType<float>() {
  return data_type::f32;
}

template <>
inline data_type toDataType<int8_t>() {
  return data_type::s8;
}

struct ConvParams {
  format_tag input_format;
  format_tag filter_format;
  format_tag output_format;

  tensor src;
  tensor weight;
  dims dst_dims;
  tensor dst;
  dims strides;
  dims dilates;
  dims padding_l;
  dims padding_r;
  int groups;
  bool weight_is_const;
};

template <typename Tinput, int N, typename Tfilter = Tinput,
          typename Toutput = Tinput>
bool parseConvParams(ExecutionContext* ctx, MemRefType<Tinput, N> input,
                     MemRefType<Tfilter, N> kernel,
                     MemRefType<int32_t, 1> padding,
                     MemRefType<Toutput, N> output,
                     MemRefType<int32_t, 1> metadata, ConvParams* params) {
  bool promote_conv1d_to_conv2d = (N == 3 && promoteConv1DToConv2D());
  int effectiveN = (promote_conv1d_to_conv2d ? N + 1 : N);

  if (promote_conv1d_to_conv2d) {
    params->padding_l.push_back(0);
    params->padding_r.push_back(0);
  }
  for (int i = 0; i < N - 2; ++i) {
    params->padding_l.push_back(padding.data[2 * i]);
    params->padding_r.push_back(padding.data[2 * i + 1]);
  }
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "input: " << input.data << ": ";
    for (int i = 0; i < N; ++i) TAO_VLOG(0) << input.sizes[i];

    TAO_VLOG(0) << "kernel: " << kernel.data << ": ";
    for (int i = 0; i < N; ++i) TAO_VLOG(0) << kernel.sizes[i];

    TAO_VLOG(0) << "output: " << output.data << ": ";
    for (int i = 0; i < N; ++i) TAO_VLOG(0) << output.sizes[i];

    TAO_VLOG(0) << "padding_l: ";
    for (int i = 0; i < params->padding_l.size(); ++i) {
      TAO_VLOG(0) << " " << params->padding_l[i];
    }
    TAO_VLOG(0) << "padding_r: ";
    for (int i = 0; i < params->padding_r.size(); ++i) {
      TAO_VLOG(0) << " " << params->padding_r[i];
    }
  }

  std::vector<char> format_buffer(effectiveN + 1, 0);
  int idx = 0;
  int ic = 0;
  dims input_dims(effectiveN, 0);
  if (promote_conv1d_to_conv2d) {
    int batch_dim = metadata.data[idx];
    int feature_dim = metadata.data[idx + 1];
    int spatial_dim = metadata.data[idx + 2];
    int extended_batch_dim =
        (batch_dim < spatial_dim) ? batch_dim : batch_dim + 1;
    int extended_feature_dim =
        (feature_dim < spatial_dim) ? feature_dim : feature_dim + 1;
    int extended_spatial_dim_0 = spatial_dim;
    int extended_spatial_dim_1 = spatial_dim + 1;

    ic = input.sizes[feature_dim];
    format_buffer[extended_batch_dim] = 'a';
    format_buffer[extended_feature_dim] = 'b';
    format_buffer[extended_spatial_dim_0] = 'c';
    format_buffer[extended_spatial_dim_1] = 'd';
    input_dims[0] = input.sizes[batch_dim];
    input_dims[1] = input.sizes[feature_dim];
    input_dims[2] = 1;
    input_dims[3] = input.sizes[spatial_dim];

    idx += 3;
  } else {
    for (int i = 0; i < N; ++i) {
      if (i == 1) ic = input.sizes[metadata.data[idx]];
      input_dims[i] = input.sizes[metadata.data[idx]];
      format_buffer[metadata.data[idx++]] = 'a' + i;
    }
  }
  params->input_format = str2format(format_buffer.data());
  if (params->input_format == format_tag::undef) {
    ctx->signalError(Context::FAILURE, "invalid input format for conv op");
    return false;
  }
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "input format: " << format_buffer.data();
  }

  int kc = kernel.sizes[metadata.data[idx]];
  dims filter_dims(effectiveN, 0);
  if (promote_conv1d_to_conv2d) {
    int input_feature_dim = metadata.data[idx];
    int output_feature_dim = metadata.data[idx + 1];
    int spatial_dim = metadata.data[idx + 2];
    int extended_input_feature_dim = (input_feature_dim < spatial_dim)
                                         ? input_feature_dim
                                         : input_feature_dim + 1;
    int extended_output_feature_dim = (output_feature_dim < spatial_dim)
                                          ? output_feature_dim
                                          : output_feature_dim + 1;
    int extended_spatial_dim_0 = spatial_dim;
    int extended_spatial_dim_1 = spatial_dim + 1;

    format_buffer[extended_output_feature_dim] = 'a';
    format_buffer[extended_input_feature_dim] = 'b';
    format_buffer[extended_spatial_dim_0] = 'c';
    format_buffer[extended_spatial_dim_1] = 'd';
    filter_dims[0] = kernel.sizes[output_feature_dim];
    filter_dims[1] = kernel.sizes[input_feature_dim];
    filter_dims[2] = 1;
    filter_dims[3] = kernel.sizes[spatial_dim];

    idx += 3;
  } else {
    filter_dims[1] = kernel.sizes[metadata.data[idx]];
    format_buffer[metadata.data[idx++]] = 'b';
    filter_dims[0] = kernel.sizes[metadata.data[idx]];
    format_buffer[metadata.data[idx++]] = 'a';
    for (int i = 2; i < N; ++i) {
      filter_dims[i] = kernel.sizes[metadata.data[idx]];
      format_buffer[metadata.data[idx++]] = 'a' + i;
    }
  }
  params->filter_format = str2format(format_buffer.data());
  if (params->filter_format == format_tag::undef) {
    ctx->signalError(Context::FAILURE, "invalid filter format for conv op");
    return false;
  }
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "filter format: " << format_buffer.data();
  }

  dims output_dims(effectiveN, 0);
  if (promote_conv1d_to_conv2d) {
    int batch_dim = metadata.data[idx];
    int feature_dim = metadata.data[idx + 1];
    int spatial_dim = metadata.data[idx + 2];
    int extended_batch_dim =
        (batch_dim < spatial_dim) ? batch_dim : batch_dim + 1;
    int extended_feature_dim =
        (feature_dim < spatial_dim) ? feature_dim : feature_dim + 1;
    int extended_spatial_dim_0 = spatial_dim;
    int extended_spatial_dim_1 = spatial_dim + 1;

    format_buffer[extended_batch_dim] = 'a';
    format_buffer[extended_feature_dim] = 'b';
    format_buffer[extended_spatial_dim_0] = 'c';
    format_buffer[extended_spatial_dim_1] = 'd';
    output_dims[0] = output.sizes[batch_dim];
    output_dims[1] = output.sizes[feature_dim];
    output_dims[2] = 1;
    output_dims[3] = output.sizes[spatial_dim];

    idx += 3;
  } else {
    for (int i = 0; i < N; ++i) {
      output_dims[i] = output.sizes[metadata.data[idx]];
      format_buffer[metadata.data[idx++]] = 'a' + i;
    }
  }

  params->output_format = str2format(format_buffer.data());
  if (params->output_format == format_tag::undef) {
    ctx->signalError(Context::FAILURE, "invalid output format for conv op");
    return false;
  }
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "output format: " << format_buffer.data();
  }

  if (promote_conv1d_to_conv2d) {
    params->strides.push_back(1);
    params->dilates.push_back(1);
  }
  for (int i = 0; i < N - 2; ++i) {
    params->strides.push_back(metadata.data[idx++]);
  }
  for (int i = 0; i < N - 2; ++i) {
    params->dilates.push_back(metadata.data[idx++]);
  }
  params->groups = ic / kc;
  params->weight_is_const = metadata.data[idx++];
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "strides: ";
    for (int i = 0; i < effectiveN - 2; ++i) {
      TAO_VLOG(0) << " " << params->strides[i];
    }
    TAO_VLOG(0) << "dilations: ";
    for (int i = 0; i < effectiveN - 2; ++i) {
      TAO_VLOG(0) << " " << params->dilates[i];
    }
    TAO_VLOG(0) << "ic = " << ic << ", kc = " << kc
                << ", groups = " << params->groups;
    TAO_VLOG(0) << "input mkl logical shape (NCHW) = ";
    for (int i = 0; i < effectiveN; ++i) {
      TAO_VLOG(0) << "\t" << input_dims[i];
    }

    TAO_VLOG(0) << "filter mkl logical shape (OIHW) = ";
    for (int i = 0; i < effectiveN; ++i) {
      TAO_VLOG(0) << "\t" << filter_dims[i];
    }

    TAO_VLOG(0) << "output mkl logical shape (NCHW) = ";
    for (int i = 0; i < effectiveN; ++i) {
      TAO_VLOG(0) << "\t" << output_dims[i];
    }
    TAO_VLOG(0) << "weight_is_const = " << params->weight_is_const;
  }

  data_type input_dtype = toDataType<Tinput>();
  if (input_dtype == data_type::undef) {
    ctx->signalError(Context::FAILURE, "invalid input dtype for conv op");
    return false;
  }
  params->src =
      tensor{input_dims, input_dtype, params->input_format, input.data};

  data_type filter_dtype = toDataType<Tfilter>();
  if (filter_dtype == data_type::undef) {
    ctx->signalError(Context::FAILURE, "invalid filter dtype for conv op");
    return false;
  }
  params->weight =
      tensor{filter_dims, filter_dtype, params->filter_format, kernel.data};

  data_type output_dtype = toDataType<Toutput>();
  if (output_dtype == data_type::undef) {
    ctx->signalError(Context::FAILURE, "invalid output dtype for conv op");
    return false;
  }
  params->dst =
      tensor{output_dims, output_dtype, params->output_format, output.data};
  params->dst_dims = output_dims;
  return true;
}

}  // namespace ral
}  // namespace tao

#endif  // defined(TAO_CPU_ONLY) && defined(TAO_ENABLE_MKLDNN)

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_MKLDNN_H_
