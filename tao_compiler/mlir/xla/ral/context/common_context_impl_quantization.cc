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

#if defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)

#include <sstream>
#include <thread>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_mkldnn.h"

namespace tao {
namespace ral {

using namespace arm_compute;

namespace {

template <int NDims>
void ral_qconv_s8_s8_s8(
    ExecutionContext* ctx, opaque_t /*stream_handle*/,
    MemRefType<int8_t, NDims> input, MemRefType<int8_t, NDims> kernel,
    MemRefType<int32_t, 1> padding, MemRefType<float, 0> inputScales,
    MemRefType<float, 1> filterScales, MemRefType<float, 0> outputScales,
    MemRefType<int8_t, NDims> output, MemRefType<int32_t, 1> metadata) {
  CpuTimer timer("ral_qconv_s8_s8_s8");
  if (isEmptyMemref(input) || isEmptyMemref(kernel) || isEmptyMemref(output)) {
    TAO_VLOG(1) << "ral_qconv_s8_s8_s8: early return for empty tensor";
    return;
  }
  ConvParams params;
  if (!parseConvParams(ctx, input, kernel, padding, output, metadata,
                       &params)) {
    ctx->signalError(Context::FAILURE, "invalid conv params");
  }

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "input scale = " << inputScales.data[0];
    TAO_VLOG(0) << "output scale = " << outputScales.data[0];
    for (int i = 0; i < filterScales.sizes[0]; ++i)
      TAO_VLOG(0) << "filter_scale[" << i << "] = " << filterScales.data[i];
  }

  if (params.groups > 1) {
    ctx->signalError(Context::FAILURE, "invalid conv params");
  }

  int N = input.sizes[0];
  int H = input.sizes[1];
  int W = input.sizes[2];
  int Ci = input.sizes[3];
  int Co = output.sizes[3];
  int Kh = kernel.sizes[1];
  int Kw = kernel.sizes[2];

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(1) << "N = " << N;
    TAO_VLOG(1) << "H = " << H;
    TAO_VLOG(1) << "W = " << W;
    TAO_VLOG(1) << "Ci = " << Ci;
    TAO_VLOG(1) << "Co = " << Co;
    TAO_VLOG(1) << "Kh = " << Kh;
    TAO_VLOG(1) << "Kw = " << Kw;
  }

  DataLayout data_layout = DataLayout::NHWC;
  TensorShape src_shape, weights_shape, biases_shape, dst_shape;
  src_shape = TensorShape(Ci, W, H, N);
  weights_shape = TensorShape(Ci, Kw, Kh, Co);
  dst_shape = TensorShape(Co, W, H, N);

  DataType data_type = DataType::QASYMM8_SIGNED;
  TensorInfo src_info = TensorInfo(src_shape, 1, data_type, data_layout);
  TensorInfo weights_info =
      TensorInfo(weights_shape, 1, DataType::QSYMM8_PER_CHANNEL, data_layout);
  TensorInfo dst_info = TensorInfo(dst_shape, 1, data_type, data_layout);

  const QuantizationInfo src_qinfo = QuantizationInfo();
  src_info.set_quantization_info(QuantizationInfo(*inputScales.data, 0));
  std::vector<float> scales(filterScales.data,
                            filterScales.data + filterScales.sizes[0]);
  weights_info.set_quantization_info(QuantizationInfo(std::move(scales)));
  dst_info.set_quantization_info(QuantizationInfo(*outputScales.data, 0));

  arm_compute::Tensor src, weights, dst;
  // Initialize tensors
  src.allocator()->init(src_info);
  weights.allocator()->init(weights_info);
  dst.allocator()->init(dst_info);

  src.allocator()->import_memory(input.data);
  weights.allocator()->import_memory(kernel.data);
  dst.allocator()->import_memory(output.data);

  NEConvolutionLayer conv{};
  conv.configure(
      &src, &weights, nullptr, &dst,
      PadStrideInfo{params.strides[1], params.strides[0], params.padding_l[1],
                    params.padding_r[1], params.padding_l[0],
                    params.padding_r[0], DimensionRoundingType::FLOOR},
      WeightsInfo{}, Size2D{params.dilates[1], params.dilates[0]});

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "params.strides[1] = " << params.strides[1];
    TAO_VLOG(0) << "params.strides[0] = " << params.strides[0];
    TAO_VLOG(0) << "params.padding_l[1] = " << params.padding_l[1];
    TAO_VLOG(0) << "params.padding_l[0] = " << params.padding_l[0];
    TAO_VLOG(0) << "params.padding_r[1] = " << params.padding_r[1];
    TAO_VLOG(0) << "params.padding_r[0] = " << params.padding_r[0];
    TAO_VLOG(0) << "params.dilates[1] = " << params.dilates[1];
    TAO_VLOG(0) << "params.dilates[0] = " << params.dilates[0];
  }

  conv.run();
}

}  // namespace

TAO_RAL_API("ral_qconv_s8_s8_s8", "cpu", ral_qconv_s8_s8_s8<4>);

}  // namespace ral
}  // namespace tao
#endif
