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

#include <array>
#include <thread>

#include "dnnl_threadpool_iface.hpp"
#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/context/mkldnn/ideep/ideep.hpp"
#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/ral_base.h"
#include "mlir/ral/ral_helper.h"

#if defined(TAO_AARCH64)
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "mlir/ral/context/common_context_impl_acl.h"
#endif

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
bool isWeightPrePackingEnabled();

// Returns true if weight prepacking for matmul is enabled.
bool isWeightPrePackingForMatMulEnabled();

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

template <>
inline data_type toDataType<bfloat16>() {
  return data_type::bf16;
}

template <>
inline data_type toDataType<float16>() {
  return data_type::f16;
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

extern const std::thread::id kDiscCpuDefaultThreadId;

struct OnednnGemmState : public Context::Resource {
  using cache_type = std::unordered_map<opaque_t, std::vector<ideep::tensor>>;

  ideep::tensor get_or_create_packed_bias(
      opaque_t bias_ptr, const ideep::tensor& bias_t,
      const dnnl::memory::desc& target_desc,
      const ideep::attr_t& target_attr = ideep::attr_t()) {
    return get_or_create_packed_data(packed_bias_cache, bias_ptr, bias_t,
                                     target_desc, target_attr);
  }

  ideep::tensor get_or_create_packed_weight(
      opaque_t weight_ptr, const ideep::tensor& weight_t,
      const dnnl::memory::desc& target_desc,
      const ideep::attr_t& target_attr = ideep::attr_t()) {
    return get_or_create_packed_data(packed_weight_cache, weight_ptr, weight_t,
                                     target_desc, target_attr);
  }

 private:
  ideep::tensor get_or_create_packed_data(
      cache_type& cache_tensor, opaque_t data_ptr, const ideep::tensor& data_t,
      const dnnl::memory::desc& target_desc,
      const ideep::attr_t& target_attr = ideep::attr_t()) {
    std::lock_guard<std::mutex> l(mu);
    ideep::tensor packed_tensor;
    auto& packed_tensors = cache_tensor[data_ptr];
    for (auto& tensor : packed_tensors) {
      if (target_desc == tensor.get_desc()) {
        packed_tensor = tensor;
        break;
      }
    }
    if (packed_tensor.is_empty()) {
      packed_tensor = data_t.reorder_if_differ_in(target_desc, target_attr);
      packed_tensors.push_back(packed_tensor);
    }
    return packed_tensor;
  }
  std::mutex mu;
  cache_type packed_weight_cache;
  cache_type packed_bias_cache;
};

struct GEMMParamsKeyHasher;

struct GEMMParamsKey {
  int m = -1;
  int n = -1;
  int k = -1;
  int batch = 1;
  bool transpose_a = false;
  bool transpose_b = false;
  opaque_t const_weight_ptr = nullptr;
  opaque_t const_bias_ptr = nullptr;
  // We need this since the matmul primitive is not thead safe.
  // To enable large parallelism, we cache the primitive per-thread.
  std::thread::id tid;

  bool operator==(const GEMMParamsKey& rhs) const {
    return m == rhs.m && n == rhs.n && k == rhs.k && batch == rhs.batch &&
           transpose_a == rhs.transpose_a && transpose_b == rhs.transpose_b &&
           const_weight_ptr == rhs.const_weight_ptr &&
           const_bias_ptr == rhs.const_bias_ptr && tid == rhs.tid;
  }

  using Hasher = GEMMParamsKeyHasher;
};

struct GEMMParamsKeyHasher {
  std::size_t operator()(const GEMMParamsKey& key) const {
    std::size_t seed = std::hash<int>()(key.m);
    hash_combine(seed, key.n);
    hash_combine(seed, key.k);
    hash_combine(seed, key.batch);
    hash_combine(seed, key.transpose_a);
    hash_combine(seed, key.transpose_b);
    hash_combine(seed, key.const_weight_ptr);
    hash_combine(seed, key.const_bias_ptr);
    hash_combine(seed, key.tid);
    return seed;
  }
};

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
inline GEMMParamsKey makeGEMMParamsKey(const MemRefType<Tinput, NDims>& input,
                                       const MemRefType<Tfilter, NDims>& weight,
                                       const MemRefType<Toutput, NDims>& result,
                                       bool tp_a, bool tp_b,
                                       bool weight_is_const,
                                       const std::thread::id& tid) {
  GEMMParamsKey key;
  key.tid = tid;
  key.const_weight_ptr = weight_is_const ? weight.data : nullptr;
  for (int i = 0; i < NDims - 2; ++i) {
    key.batch *= input.sizes[i];
  }
  key.m = tp_a ? input.sizes[NDims - 1] : input.sizes[NDims - 2];
  key.n = tp_b ? weight.sizes[NDims - 2] : weight.sizes[NDims - 1];
  key.k = tp_a ? input.sizes[NDims - 2] : input.sizes[NDims - 1];
  key.transpose_a = tp_a;
  key.transpose_b = tp_b;

  return key;
}

template <typename Tinput, typename Tbias, int NDims, int BiasDims,
          typename Tfilter = Tinput, typename Toutput = Tinput>
inline GEMMParamsKey makeGEMMWithBiasParamsKey(
    const MemRefType<Tinput, NDims>& input,
    const MemRefType<Tfilter, NDims>& weight,
    const MemRefType<Tbias, BiasDims>& bias,
    const MemRefType<Toutput, NDims>& result, bool tp_a, bool tp_b,
    bool weight_is_const, bool bias_is_const, const std::thread::id& tid) {
  auto key = makeGEMMParamsKey(input, weight, result, tp_a, tp_b,
                               weight_is_const, tid);
  key.const_bias_ptr = bias_is_const ? bias.data : nullptr;
  return key;
}

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
inline GEMMParamsKey makeDynamicShapeGEMMParamsKey(
    const MemRefType<Tinput, NDims>& input,
    const MemRefType<Tfilter, NDims>& weight,
    const MemRefType<Toutput, NDims>& result, bool tp_a, bool tp_b,
    bool weight_is_const, const std::thread::id& tid) {
  GEMMParamsKey key;
  key.tid = tid;
  key.const_weight_ptr = weight_is_const ? weight.data : nullptr;
  for (int i = 0; i < NDims - 2; ++i) {
    key.batch *= weight.sizes[i];
  }
  key.n = tp_b ? weight.sizes[NDims - 2] : weight.sizes[NDims - 1];
  key.k = tp_a ? input.sizes[NDims - 2] : input.sizes[NDims - 1];
  key.transpose_a = tp_a;
  key.transpose_b = tp_b;

  return key;
}

template <typename Tinput, typename Tbias, int NDims, int BiasDims,
          typename Tfilter = Tinput, typename Toutput = Tinput>
inline GEMMParamsKey makeDynamicShapeGEMMWithBiasParamsKey(
    const MemRefType<Tinput, NDims>& input,
    const MemRefType<Tfilter, NDims>& weight,
    const MemRefType<Tbias, BiasDims>& bias,
    const MemRefType<Toutput, NDims>& result, bool tp_a, bool tp_b,
    bool weight_is_const, bool bias_is_const, const std::thread::id& tid) {
  auto key = makeDynamicShapeGEMMParamsKey(input, weight, result, tp_a, tp_b,
                                           weight_is_const, tid);
  key.const_bias_ptr = bias_is_const ? bias.data : nullptr;
  return key;
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
  bool weight_is_const = false;
  bool is_depthwise = false;
};

struct ConvParamsKeyHasher;

struct ConvParamsKey {
  std::vector<int64_t> src_dims;
  std::vector<int64_t> weight_dims;
  std::vector<int64_t> dst_dims;
  // padding & stride & dilation & groups & weight_is_const ...
  std::vector<int32_t> metadata;
  const_buffer_t weight_ptr = nullptr;
  // We need this in case the cased kernel is not thead safe.
  // To enable large parallelism, we cache the primitive per-thread.
  std::thread::id tid;

  using Hasher = ConvParamsKeyHasher;
};

inline bool operator==(const ConvParamsKey& lhs, const ConvParamsKey& rhs) {
  return (lhs.src_dims == rhs.src_dims && lhs.weight_dims == rhs.weight_dims &&
          lhs.dst_dims == rhs.dst_dims && lhs.metadata == rhs.metadata &&
          lhs.weight_ptr == rhs.weight_ptr && lhs.tid == rhs.tid);
}

struct ConvParamsKeyHasher {
  std::size_t operator()(const ConvParamsKey& key) const {
    std::size_t seed = std::hash<size_t>()(key.src_dims.size());
    hash_combine(seed, key.weight_dims.size());
    hash_combine(seed, key.dst_dims.size());
    hash_combine(seed, key.metadata.size());
    hash_combine(seed, key.weight_ptr);
    hash_combine(seed, key.tid);
    for (size_t i = 0; i < key.src_dims.size(); ++i) {
      hash_combine(seed, key.src_dims[i]);
    }
    for (size_t i = 0; i < key.weight_dims.size(); ++i) {
      hash_combine(seed, key.weight_dims[i]);
    }
    for (size_t i = 0; i < key.dst_dims.size(); ++i) {
      hash_combine(seed, key.dst_dims[i]);
    }
    for (size_t i = 0; i < key.metadata.size(); ++i) {
      hash_combine(seed, key.metadata[i]);
    }
    return seed;
  }
};

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
inline ConvParamsKey makeConvParamsKey(const MemRefType<Tinput, NDims>& input,
                                       const MemRefType<Tfilter, NDims>& kernel,
                                       const MemRefType<int32_t, 1>& padding,
                                       const MemRefType<Toutput, NDims>& output,
                                       const MemRefType<int32_t, 1>& metadata,
                                       const std::thread::id& tid) {
  ConvParamsKey key;
  key.src_dims.reserve(NDims);
  key.weight_dims.reserve(NDims);
  key.dst_dims.reserve(NDims);
  key.metadata.reserve(metadata.sizes[0] + padding.sizes[0]);
  for (int i = 0; i < NDims; ++i) {
    key.src_dims.push_back(input.sizes[i]);
    key.weight_dims.push_back(kernel.sizes[i]);
    key.dst_dims.push_back(output.sizes[i]);
  }
  for (int i = 0; i < metadata.sizes[0]; ++i) {
    key.metadata.push_back(metadata.data[i]);
  }
  for (int i = 0; i < padding.sizes[0]; ++i) {
    key.metadata.push_back(padding.data[i]);
  }
  key.weight_ptr = kernel.data;
  key.tid = tid;
  return key;
}

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
inline ConvParamsKey makeDynamicShapeConvParamsKey(
    const MemRefType<Tinput, NDims>& input,
    const MemRefType<Tfilter, NDims>& kernel,
    const MemRefType<int32_t, 1>& padding,
    const MemRefType<Toutput, NDims>& output,
    const MemRefType<int32_t, 1>& metadata, const std::thread::id& tid) {
  ConvParamsKey key;
  key.weight_dims.reserve(NDims);
  key.metadata.reserve(metadata.sizes[0]);
  for (int i = 0; i < NDims; ++i) {
    key.weight_dims.push_back(kernel.sizes[i]);
  }
  for (int i = 0; i < metadata.sizes[0]; ++i) {
    key.metadata.push_back(metadata.data[i]);
  }
  key.weight_ptr = kernel.data;
  key.tid = tid;
  return key;
}

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

  params->is_depthwise = (filter_dims[1] == 1);

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
    TAO_VLOG(0) << "is_depthwise = " << params->is_depthwise;
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

template <typename Tinput, typename Tfilter = Tinput, typename Toutput = Tinput>
void dumpConvLikeKernelProflingInfo(const ConvParams& params, size_t nanosec,
                                    const char* message) {
  const auto& src_dims = params.src.get_dims();
  const auto& kernel_dims = params.weight.get_dims();
  const auto& dst_dims = params.dst.get_dims();

  int64_t bytes =
      static_cast<int64_t>(params.src.get_nelems()) * sizeof(Tinput) +
      static_cast<int64_t>(params.weight.get_nelems()) * sizeof(Tfilter) +
      static_cast<int64_t>(params.dst.get_nelems()) * sizeof(Toutput);

  int64_t OC = params.dst.get_dims()[1];
  // 2 * KH * KW * KIC - 1
  int64_t flops_per_output =
      2 * static_cast<int64_t>(params.weight.get_nelems()) / OC - 1;
  // #outputs * flops_per_output
  int64_t gflops =
      static_cast<int64_t>(params.dst.get_nelems()) * flops_per_output;

  std::ostringstream sout;
  sout << message << ":\n";
  sout << "  input logical NCHW shape:\n\t";
  for (const auto& d : src_dims) {
    sout << d << " ";
  }
  sout << "\n  kernel logical OIHW shape:\n\t";
  for (const auto& d : kernel_dims) {
    sout << d << " ";
  }
  sout << "\n  output logical NCHW shape:\n\t";
  for (const auto& d : dst_dims) {
    sout << d << " ";
  }
  sout << "\n  strides:\n\t";
  for (size_t i = 0; i < params.strides.size(); ++i) {
    sout << params.strides[i] << " ";
  }
  sout << "\n  dilates:\n\t";
  for (size_t i = 0; i < params.dilates.size(); ++i) {
    sout << params.dilates[i] << " ";
  }
  sout << "\n  paddings_l:\n\t";
  for (size_t i = 0; i < params.padding_l.size(); ++i) {
    sout << params.padding_l[i] << " ";
  }
  sout << "\n  paddings_r:\n\t";
  for (size_t i = 0; i < params.padding_r.size(); ++i) {
    sout << params.padding_r[i] << " ";
  }
  TAO_VLOG(0) << sout.str() << "\n roofline:\n"
              << "\tMath Ops = " << gflops << "\n"
              << "\tBytes = " << bytes << "\n"
              << "\tBandwidth = " << double(bytes) / double(nanosec) << " GB\n"
              << "\tGFLOPS = " << double(gflops) / double(nanosec) << "\n";
  TAO_VLOG(0) << "weight_is_const = " << params.weight_is_const << "\n";
  TAO_VLOG(0) << "is_depthwise = " << params.is_depthwise << "\n";
}

#if defined(TAO_AARCH64)

// Sets thread pool configurations in the first step.
bool applyACLThreadPoolConfigIfNotSet();

template <typename OpTy>
struct AclOpInfo {
  arm_compute::Tensor src;
  arm_compute::Tensor weights;
  arm_compute::Tensor dst_s32;
  arm_compute::Tensor dst;
  arm_compute::Tensor bias;
  arm_compute::NEGEMMLowpOutputStage gemmlowp_output_stage;
  OpTy op;
};

template <typename TKey, typename TValue>
using AclOpMap = std::unordered_map<TKey, TValue, typename TKey::Hasher>;

template <typename TKey, typename OpTy>
class AclOpThreadSafeInfo {
 public:
  inline std::shared_ptr<AclOpInfo<OpTy>> getOrCreate(
      const TKey& key, std::function<std::shared_ptr<AclOpInfo<OpTy>>(
                           const arm_compute::ITensorPack*)>
                           creator) {
    int tid = ThreadLocalIndex::Get();
    if (tid < kMaxAllowedThread) {
      auto it = threadLocalStateFastPath_[tid].infoMap.find(key);
      if (it != threadLocalStateFastPath_[tid].infoMap.end()) return it->second;
    }

    std::shared_ptr<AclOpInfo<OpTy>> mainInfo;
    {
      std::lock_guard<std::mutex> l(mu);
      if (tid >= kMaxAllowedThread) {
        auto& state = threadLocalStateSlowPath_[tid];
        auto it = state.infoMap.find(key);
        if (it != state.infoMap.end()) return it->second;
      }
      auto mainInfoIt = mainInfoMap_.find(key);
      if (mainInfoIt != mainInfoMap_.end()) {
        mainInfo = mainInfoIt->second;
      }
    }

    if (!mainInfo) {
      mainInfo = creator(nullptr);
      std::string md5 = mainInfo->op.get_md5_for_packed_weight();
      std::lock_guard<std::mutex> l(mu);
      auto r = md5MainInfoMap_.emplace(md5, mainInfo);
      if (!r.second) {
        // re-use packed weight from other shape configuration.
        mainInfo = r.first->second;
        mainInfoMap_.emplace(key, mainInfo);
      } else {
        // early-return if it's the first one that has md5 as `md5`
        mainInfoMap_.emplace(key, mainInfo);
        auto& threadLocalState = (tid < kMaxAllowedThread)
                                     ? threadLocalStateFastPath_[tid]
                                     : threadLocalStateSlowPath_[tid];
        threadLocalState.infoMap.emplace(key, mainInfo);
        return mainInfo;
      }
    }

    auto info = creator(&mainInfo->op.get_packed_weight());
    std::lock_guard<std::mutex> l(mu);
    auto& threadLocalState = (tid < kMaxAllowedThread)
                                 ? threadLocalStateFastPath_[tid]
                                 : threadLocalStateSlowPath_[tid];
    threadLocalState.infoMap.emplace(key, info);
    return info;
  }

 private:
  std::mutex mu;
  struct ThreadLocalItem {
    AclOpMap<TKey, std::shared_ptr<AclOpInfo<OpTy>>> infoMap;
  };
  static constexpr const int kMaxAllowedThread = 4096;
  std::array<ThreadLocalItem, kMaxAllowedThread> threadLocalStateFastPath_;
  std::unordered_map<int, ThreadLocalItem> threadLocalStateSlowPath_;
  std::unordered_map<std::string, std::shared_ptr<AclOpInfo<OpTy>>>
      md5MainInfoMap_;
  AclOpMap<TKey, std::shared_ptr<AclOpInfo<OpTy>>> mainInfoMap_;
};

template <typename TKey, typename OpTy>
class AclOpThreadSafeInfoV2 {
 public:
  inline std::shared_ptr<AclOpInfo<OpTy>> getOrCreate(
      const TKey& key, std::function<std::shared_ptr<AclOpInfo<OpTy>>(
                           const arm_compute::ITensorPack*)>
                           creator) {
    {
      std::lock_guard<std::mutex> l(mu);
      auto mainInfoIt = mainInfoMap_.find(key);
      if (mainInfoIt != mainInfoMap_.end()) return mainInfoIt->second;
    }

    std::shared_ptr<AclOpInfo<OpTy>> mainInfo = creator(nullptr);
    std::string md5 = mainInfo->op.get_md5_for_packed_weight();

    {
      std::lock_guard<std::mutex> l(mu);
      // double-check in case other threads have inserted the items.
      auto mainInfoIt = mainInfoMap_.find(key);
      if (mainInfoIt != mainInfoMap_.end()) return mainInfoIt->second;

      auto r = md5MainInfoMap_.emplace(md5, mainInfo);
      if (!r.second) {
        // re-use packed weight from other shape configuration.
        mainInfo = r.first->second;
      } else {
        // early-return if it's the first one that has md5 as `md5`
        mainInfoMap_.emplace(key, mainInfo);
        return mainInfo;
      }
    }

    auto info = creator(&mainInfo->op.get_packed_weight());
    std::lock_guard<std::mutex> l(mu);
    auto it = mainInfoMap_.emplace(key, info);
    return it.first->second;
  }

 private:
  std::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<AclOpInfo<OpTy>>>
      md5MainInfoMap_;
  AclOpMap<TKey, std::shared_ptr<AclOpInfo<OpTy>>> mainInfoMap_;
};

template <typename TKey, typename OpTy>
using AclOpCache =
    ideep::utils::lru_cache<TKey, std::shared_ptr<AclOpInfo<OpTy>>, AclOpMap>;

template <typename TKey, typename OpTy>
struct AclOpState : public Context::Resource {
  std::mutex mu;
  AclOpCache<TKey, OpTy> cache{getWeightPrePackingCacheCapacity()};

  inline std::shared_ptr<AclOpInfo<OpTy>> getOrCreate(
      const TKey& key,
      std::function<std::shared_ptr<AclOpInfo<OpTy>>()> creator) {
    std::lock_guard<std::mutex> l(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
      it = cache.insert(std::make_pair(key, creator())).first;
    }
    return it->second;
  }
};

template <typename TKey, typename OpTy,
          typename ThreadSafeInfo = AclOpThreadSafeInfo<TKey, OpTy>>
struct AclOpThreadSafeState : public Context::Resource {
  // some helper type aliases
  using ThreadSafeInfoPtr = std::shared_ptr<ThreadSafeInfo>;
  using AclOpThreadSafeCache =
      ideep::utils::lru_cache<TKey, ThreadSafeInfoPtr, AclOpMap>;

  std::mutex mu;
  AclOpThreadSafeCache cache{getWeightPrePackingCacheCapacity()};

  inline ThreadSafeInfoPtr getOrCreate(const TKey& key) {
    std::lock_guard<std::mutex> l(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
      it = cache
               .insert(
                   std::make_pair(key, ThreadSafeInfoPtr(new ThreadSafeInfo)))
               .first;
    }
    return it->second;
  }
};

using AclDepthwiseConvInfo =
    AclOpInfo<arm_compute::DISCNEDepthwiseConvolutionLayer>;
using AclDepthwiseConvThreadSafeInfo =
    AclOpThreadSafeInfoV2<ConvParamsKey,
                          arm_compute::DISCNEDepthwiseConvolutionLayer>;
using AclDepthwiseConvState = AclOpThreadSafeState<
    ConvParamsKey, arm_compute::DISCNEDepthwiseConvolutionLayer,
    AclOpThreadSafeInfoV2<ConvParamsKey,
                          arm_compute::DISCNEDepthwiseConvolutionLayer>>;

using AclConvInfo = AclOpInfo<arm_compute::DISCNEConvolutionLayer>;
using AclConvThreadSafeInfo =
    AclOpThreadSafeInfo<ConvParamsKey, arm_compute::DISCNEConvolutionLayer>;
using AclConvState =
    AclOpThreadSafeState<ConvParamsKey, arm_compute::DISCNEConvolutionLayer>;

using AclConvThreadSafeInfoV2 =
    AclOpThreadSafeInfoV2<ConvParamsKey, arm_compute::DISCNEConvolutionLayer>;
using AclConvStateV2 =
    AclOpThreadSafeState<ConvParamsKey, arm_compute::DISCNEConvolutionLayer,
                         AclConvThreadSafeInfoV2>;

using AclQGemmInfo = AclOpInfo<arm_compute::DISCNEGEMMLowpMatrixMultiplyCore>;
using AclQGemmThreadSafeInfo =
    AclOpThreadSafeInfo<GEMMParamsKey,
                        arm_compute::DISCNEGEMMLowpMatrixMultiplyCore>;
using AclQGemmState =
    AclOpThreadSafeState<GEMMParamsKey,
                         arm_compute::DISCNEGEMMLowpMatrixMultiplyCore>;

using AclGemmInfo = AclOpInfo<arm_compute::DISCNEGEMM>;
using AclGemmThreadSafeInfo =
    AclOpThreadSafeInfo<GEMMParamsKey, arm_compute::DISCNEGEMM>;
using AclGemmState =
    AclOpThreadSafeState<GEMMParamsKey, arm_compute::DISCNEGEMM>;

#endif  // TAO_AARCH64

}  // namespace ral
}  // namespace tao

#endif  // defined(TAO_CPU_ONLY) && defined(TAO_ENABLE_MKLDNN)

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_MKLDNN_H_
