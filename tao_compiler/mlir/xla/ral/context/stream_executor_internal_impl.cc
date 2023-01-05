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

#include <dlfcn.h>

#include <functional>
#include <iostream>

#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/pdll_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_base.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"
#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
#include "bladnn/bladnn.h"
#endif

#ifdef TAO_RAL_USE_STREAM_EXECUTOR

namespace tao {
namespace ral {
namespace gpu {

namespace se = ::stream_executor;

namespace se_impl {

using namespace tensorflow;

namespace gpu_conv_impl {
#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
// layout:
// input: NHWC
// kernel: OHWI
// output: NHWC
template <typename T, int NDims>
MemRefType<T, NDims> ral_conv_biasadd(ExecutionContext* ctx,
                                      void* stream_handle /*stream_handle*/,
                                      MemRefType<T, NDims> input,
                                      MemRefType<T, NDims> kernel,
                                      MemRefType<T, 1> bias,
                                      void* customAttrs) {
  // const std::vector<int32_t> nhwc_ohwi_layout = {0, 3, 1, 2, 3, 0,
  //                                             1, 2, 0, 3, 1, 2};
  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_conv_biasadd");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  auto& stride_attr =
      dictAttr.get("stride").template as<DenseElementsPDLAttr>();
  auto stride = stride_attr.getValue<int64_t>();
  auto& padding_attr =
      dictAttr.get("padding").template as<DenseElementsPDLAttr>();
  auto padding = padding_attr.getValue<int64_t>();
  auto& dilation_attr =
      dictAttr.get("dilation").template as<DenseElementsPDLAttr>();
  auto dilation = dilation_attr.getValue<int64_t>();
  int groups = dictAttr.get("groups").template as<IntPDLAttr>().getValue();

  std::string format =
      dictAttr.get("data_format").template as<StrPDLAttr>().getValue();
  TAO_CHECK(format == "NHWC");
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream =
      static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));
  void* s = stream->implementation()->GpuStreamHack();
  int32_t n = input.sizes[0];
  T* a_data = input.data;
  T* b_data = kernel.data;
  T* c_data = bias.data;
  int32_t ic = 0;
  int32_t oc = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t oh = 0;
  int32_t ow = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t ki = 0;
  int32_t ko = 0;
  ih = input.sizes[1];
  iw = input.sizes[2];
  ic = input.sizes[3];
  ko = kernel.sizes[0];
  kh = kernel.sizes[1];
  kw = kernel.sizes[2];
  ki = kernel.sizes[3];
  oc = ko;

  int pad_h = padding[0];
  int pad_w = padding[1];
  int stride_h = stride[0];
  int stride_w = stride[1];
  int dilation_h = dilation[0];
  int dilation_w = dilation[1];

  oh = ((ih + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1);
  ow = ((iw + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1);

  int64_t resultSizes[4] = {n, oh, ow, oc};

  if (isEmptyMemref(input) || isEmptyMemref(kernel)) {
    TAO_VLOG(1) << "ral_conv_biasadd: early return for empty tensor";
    return assignMemRef<T, NDims>(nullptr, resultSizes);
    ;
  }

  auto data =
      static_cast<T*>(gpu_driver->alloc(ctx, n * oh * ow * oc * sizeof(T)));
  auto result = assignMemRef<T, NDims>(data, resultSizes);

  bool is_depthwise = false;
  if (ic != ki) {
    TAO_CHECK(ki == 1);
    is_depthwise = true;
    TAO_CHECK(groups = ic);
  }
  auto conv_kind = bladnn::ConvKind::kFprop;
  auto data_layout = bladnn::Layout::kNHWC;
  auto kernel_layout = bladnn::Layout::kNHWC;
  const float alpha = 1.0f;
  const float beta = 1.0f;
  bladnn::Dtype dtype = toBlaDNNDtype<T>();
  bool ret = false;

  ret = bladnn::conv2d(
      s, dtype, dtype, conv_kind, data_layout, kernel_layout, n, ih, iw, ic, ko,
      kh, kw, oh, ow, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      groups, &alpha, a_data, b_data, &beta, c_data, result.data, nullptr);
  if (!ret) {
    ctx->signalError(Context::FAILURE, "bladnn fail");
  }
  return result;
}
#endif

}  // namespace gpu_conv_impl

////////////////////////////////////////////////////////////////////////
///////////////           GpuConvImpl Finish
///////////////
////////////////////////////////////////////////////////////////////////

namespace groupnorm_impl {

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)

// Pre-requirement: input has layout NHWC
template <typename T>
MemRefType<T, 4> bladnn_groupnorm(ExecutionContext* ctx, void* stream_handle,
                                  MemRefType<T, 4> input,
                                  MemRefType<T, 1> weight,
                                  MemRefType<T, 1> bias, void* customAttrs) {
  size_t nElems = Size(input);
  auto driver = ctx->getDriver<gpu::GPUDriver>(gpu::GPUDriver::name());
  TAO_CHECK(driver);
  auto ptr = static_cast<T*>(driver->alloc(ctx, nElems * sizeof(T)));
  auto output = assignMemRef<T, 4>(ptr, input.sizes);

  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_groupnorm");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  auto num_group =
      dictAttr.get("num_group").template as<IntPDLAttr>().getValue();
  auto eps = dictAttr.get("eps").template as<FloatPDLAttr>().getValue();
  auto use_silu = dictAttr.get("silu").template as<BoolPDLAttr>().getValue();
  auto stream = driver->asCUStream(ctx, stream_handle);
  bool ret =
      bladnn::groupnorm(output.data, input.data, weight.data, bias.data,
                        input.sizes[0], input.sizes[1], input.sizes[2],
                        input.sizes[3], num_group, use_silu, eps, stream);
  if (!ret) {
    ctx->signalError(Context::FAILURE, "fail to call bladnn::groupnorm\n");
  }
  return output;
}

#endif

}  // namespace groupnorm_impl

}  // namespace se_impl
}  // namespace gpu

// conv ops

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
TAO_RAL_API("ral_pdll_conv_bias", "gpu",
            gpu::se_impl::gpu_conv_impl::ral_conv_biasadd<Eigen::half, 4>);
TAO_RAL_API("ral_pdll_conv_bias", "gpu",
            gpu::se_impl::gpu_conv_impl::ral_conv_biasadd<float, 4>);
TAO_RAL_API("ral_pdll_group_norm", "gpu",
            gpu::se_impl::groupnorm_impl::bladnn_groupnorm<Eigen::half>);
#endif

}  // namespace ral
}  // namespace tao

#endif
