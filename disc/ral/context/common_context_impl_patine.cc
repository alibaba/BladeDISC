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

#if defined(TAO_CPU_ONLY) && defined(TAO_ENABLE_PATINE)

#include "patine/client/interface.h"
#include "ral/context/common_context_impl.h"
#include "ral/context/context_util.h"
#include "ral/device/cpu/cpu_driver.h"
#include "ral/ral_base.h"
#include "ral/ral_helper.h"

namespace tao {
namespace ral {

template <typename T, int N>
int64_t GetBatchSize(MemRefType<T, N> memref) {
  int64_t batch = 1;
  for (int64_t i = 0; i < N - 2; ++i) {
    batch *= memref.sizes[i];
  }
  return batch;
}

template <typename T, int N, typename E = float>
void ral_batch_gemm(ExecutionContext* ctx, void* stream_handle,
                    MemRefType<T, N> A, MemRefType<T, N> B, MemRefType<T, N> C,
                    bool tp_a, bool tp_b) {
#ifndef PLATFORM_ALIBABA
  ctx->signalError(Context::FAILURE,
                   "ral_batch_gemm has no implementation yet, which"
                   " is on the road.");
#endif
  CpuTimer timer("ral_cpu_batch_gemm");
  if (isEmptyMemref(A) || isEmptyMemref(B) || isEmptyMemref(C)) {
    ctx->signalError(Context::FAILURE, "ral_batch_gemm input error");
    return;
  }

  // It would be better to use `static_assert` here while we need to support
  // lower gcc version in tao bridge ATM which does not support this well
  assert((N > 2) && "batch gemm requires operands with rank higher than 2");
  int64_t batch_a = GetBatchSize(A);
  int64_t batch_b = GetBatchSize(B);
  int64_t batch_c = GetBatchSize(C);

  if (batch_a != batch_b || batch_a != batch_c) {
    ctx->signalError(Context::FAILURE, "mismatch batch size");
    return;
  }

  char transa = tp_a ? 'T' : 'N';
  char transb = tp_b ? 'T' : 'N';
  long long m = tp_a ? A.sizes[N - 1] : A.sizes[N - 2];
  assert(C.sizes[N - 2] == m);
  long long n = tp_b ? B.sizes[N - 2] : B.sizes[N - 1];
  assert(C.sizes[N - 1] == n);
  long long k = tp_a ? A.sizes[N - 2] : A.sizes[N - 1];
  long long kb = tp_b ? B.sizes[N - 1] : B.sizes[N - 2];
  assert(kb == k);
  const T* pa = A.data;
  const T* pb = B.data;
  T* pc = C.data;
#ifdef PLATFORM_ALIBABA
  patine::client::batch_gemm(pa, pb, pc, batch_a, m, n, k, tp_a, tp_b);
#endif
  timer.Stop();

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "ral_cpu_batch_gemm:\n"
                << "\tpa = " << pa << "\n"
                << "\tpb = " << pb << "\n"
                << "\tpc = " << pc << "\n"
                << "\tbatch = " << batch_a << "\n"
                << "\tm = " << m << "\n"
                << "\tn = " << n << "\n"
                << "\tk = " << k << "\n"
                << "\ttp_a = " << tp_a << "\n"
                << "\ttp_b = " << tp_b << "\n"
                << "\tMath Ops = " << 2 * batch_a * m * n * k << "\n"
                << "\tBytes = " << sizeof(T) * batch_a * (m * n + n * k + m * k)
                << "\n"
                << "\tBandwidth = "
                << double(sizeof(T) * batch_a * (m * n + n * k + m * k)) /
                       double(timer.GetNanoSeconds())
                << " GB\n"
                << "\tGFLOPS = "
                << double(2 * batch_a * m * n * k) /
                       double(timer.GetNanoSeconds())
                << "\n";
  }
}

TAO_RAL_API("ral_gemm", "cpu", ral_batch_gemm<float, 3>);
TAO_RAL_API("ral_gemm", "cpu", ral_batch_gemm<float, 4>);

template <typename T, typename E = float>
void ral_gemm(ExecutionContext* ctx, void* stream_handle, MemRefType<T, 2> A,
              MemRefType<T, 2> B, MemRefType<T, 2> C, bool tp_a, bool tp_b) {
#ifndef PLATFORM_ALIBABA
  ctx->signalError(Context::FAILURE,
                   "ral_batch_gemm has no implementation yet, which"
                   " is on the road.");
#endif
  CpuTimer timer("ral_cpu_gemm");
  long long lda = A.strides[0];
  long long ldb = B.strides[0];
  long long ldc = C.strides[0];
  long long m = tp_a ? A.sizes[1] : A.sizes[0];
  long long n = tp_b ? B.sizes[0] : B.sizes[1];
  long long k = tp_a ? A.sizes[0] : A.sizes[1];
  T* pa = A.data;
  T* pb = B.data;
  T* pc = C.data;
  assert((tp_b ? B.sizes[1] : B.sizes[0]) == k);
  assert(C.sizes[0] == m);
  assert(C.sizes[1] == n);
  char ta = tp_a ? 'T' : 'N';
  char tb = tp_b ? 'T' : 'N';
#ifdef PLATFORM_ALIBABA
  patine::client::gemm(pa, pb, pc, m, n, k, tp_a, tp_b);
#endif
  timer.Stop();

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "ral_cpu_gemm:\n"
                << "\tpa = " << pa << "\n"
                << "\tpb = " << pb << "\n"
                << "\tpc = " << pc << "\n"
                << "\tm = " << m << "\n"
                << "\tn = " << n << "\n"
                << "\tk = " << k << "\n"
                << "\ttp_a = " << tp_a << "\n"
                << "\ttp_b = " << tp_b << "\n"
                << "\tMath Ops = " << 2 * m * n * k << "\n"
                << "\tBytes = " << sizeof(T) * (m * n + n * k + m * k) << "\n"
                << "\tBandwidth = "
                << double(sizeof(T) * (m * n + n * k + m * k)) /
                       double(timer.GetNanoSeconds())
                << " GB\n"
                << "\tGFLOPS = "
                << double(2 * m * n * k) / double(timer.GetNanoSeconds())
                << "\n";
  }
}

TAO_RAL_API("ral_gemm", "cpu", ral_gemm<float>);

bool layout_match(const std::vector<int32_t>& ref,
                  MemRefType<int32_t, 1> metadata) {
  if (ref.size() > metadata.sizes[0]) {
    return false;
  }
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != metadata.data[i]) {
      return false;
    }
  }
  return true;
}

template <typename T, int N>
void ral_conv(ExecutionContext* ctx, void* stream_handle,
              MemRefType<T, N> input, MemRefType<T, N> kernel,
              MemRefType<int32_t, 1> padding, MemRefType<T, N> output,
              MemRefType<int32_t, 1> metadata) {
#ifndef PLATFORM_ALIBABA
  ctx->signalError(Context::FAILURE,
                   "ral_conv has no implementation yet, which"
                   " is on the road.");
#endif
  static_assert(N == 4, "Only conv2d supported for now");
  CpuTimer timer("ral_cpu_conv");
  if (isEmptyMemref(input) || isEmptyMemref(kernel) || isEmptyMemref(output)) {
    TAO_VLOG(1) << "ral_conv: early return for empty tensor";
    return;
  }

  if (TAO_VLOG_IS_ON(2)) {
    print_memref(input, "input");
    print_memref(kernel, "kernel");
    print_memref(padding, "padding");
    for (int i = 0; i < N - 2; ++i) {
      TAO_VLOG(0) << "\tpadding for dim #" << i << ": (" << padding.data[2 * i]
                  << ", " << padding.data[2 * i + 1] << ")";
    }
    print_memref(output, "output");
    print_memref(metadata, "metadata");
    for (int i = 0; i < 5 * N - 4; ++i) {
      TAO_VLOG(0) << "\t#" << i << ": " << metadata.data[i];
    }
  }
  int32_t n = input.sizes[0];
  assert(n == output.sizes[0]);
  T* a_data = input.data;
  T* b_data = kernel.data;
  T* c_data = output.data;
  const std::vector<int32_t> nchw_oihw_layout = {0, 1, 2, 3, 1, 0,
                                                 2, 3, 0, 1, 2, 3};
  const std::vector<int32_t> nhwc_hwio_layout = {0, 3, 1, 2, 2, 3,
                                                 0, 1, 0, 3, 1, 2};
  const char* data_layout;
  const char* kernel_layout;
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
  if (layout_match(nchw_oihw_layout, metadata)) {
    data_layout = "NCHW";
    kernel_layout = "OIHW";
    ic = input.sizes[1];
    ih = input.sizes[2];
    iw = input.sizes[3];
    ko = kernel.sizes[0];
    ki = kernel.sizes[1];
    kh = kernel.sizes[2];
    kw = kernel.sizes[3];
    oc = output.sizes[1];
    oh = output.sizes[2];
    ow = output.sizes[3];
  } else if (layout_match(nhwc_hwio_layout, metadata)) {
    data_layout = "NHWC";
    kernel_layout = "HWIO";
    ih = input.sizes[1];
    iw = input.sizes[2];
    ic = input.sizes[3];
    kh = kernel.sizes[0];
    kw = kernel.sizes[1];
    ki = kernel.sizes[2];
    ko = kernel.sizes[3];
    oh = output.sizes[1];
    ow = output.sizes[2];
    oc == output.sizes[3];
  } else {
    ctx->signalError(Context::FAILURE, "layout not supported");
    return;
  }
  assert(ko == oc);
  bool is_depthwise = false;
  int32_t groups = 1;
  if (ic != ki) {
    assert(ki == 1);
    is_depthwise = true;
    groups = ic;
  }

  std::vector<int32_t> pad;
  for (int i = 0; i < 4; ++i) {
    pad.push_back(padding.data[i]);
  }
  size_t offset = N * 3;
  int32_t sh = metadata.data[offset++];
  int32_t sw = metadata.data[offset++];
  int32_t dh = metadata.data[offset++];
  int32_t dw = metadata.data[offset++];
#ifdef PLATFORM_ALIBABA
  if (is_depthwise) {
    patine::client::depthwise_conv2d(a_data, b_data, c_data, n, ic, ih, iw, oc,
                                     kh, kw, oh, ow, data_layout, kernel_layout,
                                     pad, sh, sw, dh, dw, false);
  } else {
    patine::client::conv2d(a_data, b_data, c_data, n, ic, ih, iw, oc, kh, kw,
                           oh, ow, data_layout, kernel_layout, groups, pad, sh,
                           sw, dh, dw, false);
  }
#endif
}

TAO_RAL_API("ral_conv", "cpu", ral_conv<float, 4>);

}  // namespace ral
}  // namespace tao
#endif
