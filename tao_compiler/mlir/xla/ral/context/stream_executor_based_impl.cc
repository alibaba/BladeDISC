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

#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"

#include <functional>
#include <iostream>

#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
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

////////////////////////////////////////////////////////////////////////
///////////////           GpuGemmImpl Begin
///////////////
////////////////////////////////////////////////////////////////////////

struct GemmTuningCacheKey {
  int64_t m;
  int64_t n;
  int64_t k;
  bool lhs_transpose;
  bool rhs_transpose;
  bool o_transpose;
  bool operator<(const GemmTuningCacheKey& other) const {
    if (m != other.m) {
      return (m < other.m);
    } else if (n != other.n) {
      return (n < other.n);
    } else if (k != other.k) {
      return (k < other.k);
    } else if (lhs_transpose != other.lhs_transpose) {
      return (other.lhs_transpose);
    } else if (rhs_transpose != other.rhs_transpose) {
      return (other.rhs_transpose);
    } else if (o_transpose != other.o_transpose) {
      return (other.o_transpose);
    } else {
      return false;
    }
  }
};

GemmTuningCacheKey makeGemmTuningCacheKey(int64_t lhs_size0, int64_t lhs_size1,
                                          bool lhs_transpose, int64_t rhs_size0,
                                          int64_t rhs_size1,
                                          bool rhs_transpose) {
  GemmTuningCacheKey key;
  key.lhs_transpose = lhs_transpose;
  key.rhs_transpose = rhs_transpose;
  key.o_transpose = false;
  key.m = key.lhs_transpose ? lhs_size1 : lhs_size0;
  key.n = key.rhs_transpose ? rhs_size0 : rhs_size1;
  key.k = key.lhs_transpose ? lhs_size0 : lhs_size1;
  return key;
}

struct MatrixDescriptor {
  se::DeviceMemoryBase data;
  bool transpose;
  int64_t num_rows;
  int64_t num_cols;
  int64_t batch;
};

template <typename Element>
inline MatrixDescriptor makeMatrixDescriptor(ExecutionContext* ctx,
                                             Element* data, int64_t size0,
                                             int64_t size1, bool transpose,
                                             int64_t batch = 1) {
  se::DeviceMemoryBase device_memory((void*)data,
                                     size0 * size1 * sizeof(Element));
  int64_t num_rows = size0;
  int64_t num_cols = size1;
  return MatrixDescriptor{device_memory, transpose, num_rows, num_cols, batch};
};

template <typename NativeT>
inline se::blas::ComputationType NativeTypeToBlasType() {
  LOG(FATAL) << "Unsupported type.";
  return se::blas::ComputationType::kF32;
}

template <>
inline se::blas::ComputationType NativeTypeToBlasType<float>() {
  return se::blas::ComputationType::kF32;
}

template <>
inline se::blas::ComputationType NativeTypeToBlasType<Eigen::half>() {
  // TODO(disc): figure out why master XLA use ComputationType::kF32 for half
  // type.
  return se::blas::ComputationType::kF32;
}

template <>
inline se::blas::ComputationType NativeTypeToBlasType<double>() {
  return se::blas::ComputationType::kF64;
}

// The template was introduced, because not all instantiation of
// DoGemmWithAlgorithm template arguments was support by ThenBlasGemv.
template <typename InT, typename OutT, typename AlphaBeta>
inline bool TrySgemvInternal(se::Stream* stream, se::blas::Transpose trans,
                             uint64 m, uint64 n, AlphaBeta alpha,
                             const se::DeviceMemory<InT>& a, int lda,
                             const se::DeviceMemory<InT>& x, int incx,
                             AlphaBeta beta, se::DeviceMemory<OutT>* y,
                             int incy) {
  return true;
}

// Currently, we only support instantiation <float, float, float>
template <>
inline bool TrySgemvInternal<float, float, float>(
    se::Stream* stream, se::blas::Transpose trans, uint64 m, uint64 n,
    float alpha, const se::DeviceMemory<float>& a, int lda,
    const se::DeviceMemory<float>& x, int incx, float beta,
    se::DeviceMemory<float>* y, int incy) {
  return stream
      ->ThenBlasGemv(trans, m, n,
                     /*alpha=*/alpha, a,
                     /*leading dim of RHS=*/lda, x,
                     /*incx*/ incx, /*beta=*/beta, y, /*incy*/ incy)
      .ok();
}

template <typename InT, typename OutT, typename AlphaBeta>
static bool DoGemmWithAlgorithm(
    int64_t batch_size, MatrixDescriptor lhs_matrix,
    MatrixDescriptor rhs_matrix, MatrixDescriptor output_matrix,
    AlphaBeta alpha, AlphaBeta beta, se::Stream* stream,
    absl::optional<se::blas::AlgorithmType> algorithm,
    se::blas::ProfileResult* output_profile_result) {
  DCHECK(!output_matrix.transpose);

  se::blas::ComputationType computation_type = NativeTypeToBlasType<OutT>();
  se::DeviceMemory<InT> lhs_data(lhs_matrix.data);
  se::DeviceMemory<InT> rhs_data(rhs_matrix.data);
  se::DeviceMemory<OutT> output_data(output_matrix.data);

  auto lhs_transpose = lhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto rhs_transpose = rhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto m = lhs_matrix.transpose ? lhs_matrix.num_cols : lhs_matrix.num_rows;
  auto k = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;
  auto n = rhs_matrix.transpose ? rhs_matrix.num_rows : rhs_matrix.num_cols;
  if (std::is_same<InT, float>::value && std::is_same<OutT, float>::value &&
      std::is_same<AlphaBeta, float>::value && m == 1 && batch_size == 1) {
    return TrySgemvInternal<InT, OutT, AlphaBeta>(
        stream, rhs_transpose, rhs_matrix.num_cols, rhs_matrix.num_rows,
        /*alpha=*/alpha, rhs_data,
        /*leading dim of RHS=*/rhs_matrix.num_cols, lhs_data,
        /*incx*/ 1, /*beta=*/beta, &output_data, /*incy*/ 1);
  }
  if (algorithm) {
    // Autotuning is disabled for batch_size != 1.
    CHECK_EQ(1, batch_size);
    /*
     * Since cublas describes matrix in col major,
     * Thus we perform B' x A' = C'
     */
    return stream
        ->ThenBlasGemmWithAlgorithm(
            rhs_transpose, lhs_transpose, n, m,
            /*size of reduce dim=*/k,
            /*alpha=*/static_cast<InT>(alpha), rhs_data,
            /*leading dim of RHS=*/rhs_matrix.num_cols, lhs_data,
            /*leading dim of LHS=*/lhs_matrix.num_cols,
            /*beta=*/static_cast<OutT>(beta), &output_data,
            /*leading dim of output=*/n, computation_type, *algorithm,
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 11)
            se::blas::kDefaultComputePrecision,
#endif
            output_profile_result)
        .ok();
  }

  // TODO: need check accuracy of batched matmul
  if (batch_size != 1) {
    int64_t lhs_stride = lhs_matrix.num_rows * lhs_matrix.num_cols;
    int64_t rhs_stride = rhs_matrix.num_rows * rhs_matrix.num_cols;
    int64_t output_stride = output_matrix.num_rows * output_matrix.num_cols;
    return stream
        ->ThenBlasGemmStridedBatched(
            rhs_transpose, lhs_transpose, n, m, /*size of reduce dim=*/k,
            /*alpha=*/static_cast<AlphaBeta>(alpha), rhs_data,
            /*leading dim of RHS=*/rhs_matrix.num_cols, rhs_stride, lhs_data,
            /*leading dim of LHS=*/lhs_matrix.num_cols, lhs_stride,
            /*beta=*/static_cast<AlphaBeta>(beta), &output_data,
            /*leading dim of output=*/n, output_stride, batch_size
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 11)
            ,
            se::blas::kDefaultComputePrecision
#endif
            )
        .ok();
  }

  return stream
      ->ThenBlasGemm(rhs_transpose, lhs_transpose, n, m,
                     /*size of reduce dim=*/k,
                     /*alpha=*/static_cast<AlphaBeta>(alpha), rhs_data,
                     /*leading dim of RHS=*/rhs_matrix.num_cols, lhs_data,
                     /*leading dim of LHS=*/lhs_matrix.num_cols,
                     /*beta=*/static_cast<AlphaBeta>(beta), &output_data,
                     /*leading dim of output=*/n)
      .ok();
}

// gemm_algorithm_pick also implemented correctness check, which
// is omitted for simplicity
template <typename InT, typename OutT, typename AlphaBeta>
se::blas::AlgorithmType tuningGemm(se::Stream* stream,
                                   MatrixDescriptor lhs_matrix,
                                   MatrixDescriptor rhs_matrix,
                                   MatrixDescriptor output_matrix) {
  std::vector<se::blas::AlgorithmType> algorithms;
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 10 || TF_MAJOR_VERSION > 2
  CHECK(stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms));
#else
  CHECK(stream->parent()->GetBlasGemmAlgorithms(&algorithms));
#endif
  float best_time = std::numeric_limits<float>::infinity();
  se::blas::AlgorithmType best_algo = algorithms.front();
  for (se::blas::AlgorithmType algorithm : algorithms) {
    se::blas::ProfileResult profile_result;
    DoGemmWithAlgorithm<InT, OutT, AlphaBeta>(
        /*batch_size*/ 1, lhs_matrix, rhs_matrix, output_matrix,
        /*alpha*/ 1., /*beta*/ 0., stream, algorithm, &profile_result);

    if (!profile_result.is_valid()) {
      TAO_VLOG(1) << "algo: " << algorithm << " is invalid.";
      // Unsupported algorithm.
      continue;
    }

    TAO_VLOG(1) << "algo: " << algorithm << "take "
                << profile_result.elapsed_time_in_ms();

    if (profile_result.elapsed_time_in_ms() < best_time) {
      best_time = profile_result.elapsed_time_in_ms();
      best_algo = algorithm;
    }
  }
  TAO_VLOG(1) << "tuned best algo: " << best_algo;
  return best_algo;
}

struct RalGemmState : public Context::Resource {
  std::mutex mu;
  std::map<GemmTuningCacheKey, se::blas::AlgorithmType> gemm_tuning_cache;
};

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
template <typename T>
bladnn::Dtype toBlaDNNDtype() {
  if (std::is_same<T, Eigen::half>::value) {
    return bladnn::Dtype::kF16;
  }
  if (std::is_same<T, float>::value) {
    return bladnn::Dtype::kF32;
  }
  if (std::is_same<T, double>::value) {
    return bladnn::Dtype::kF64;
  }
  return bladnn::Dtype::kUnknown;
}
#endif

template <typename InT, typename OutT, typename E = float>
void ral_gemm(ExecutionContext* ctx, void* stream_handle, MemRefType<InT, 2> A,
              MemRefType<InT, 2> B, MemRefType<OutT, 2> C, bool tp_a, bool tp_b,
              bool weight_is_const) {
  if (isEmptyMemref(A) || isEmptyMemref(B) || isEmptyMemref(C)) {
    TAO_VLOG(1) << "ral_gemm: early return for empty tensor";
    return;
  }

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  {
    auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
    auto stream =
        static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));
    void* s = stream->implementation()->GpuStreamHack();
    bladnn::Context bladnn_ctx{s};
    bladnn::Dtype in_dtype = toBlaDNNDtype<InT>();
    bladnn::Dtype out_dtype = toBlaDNNDtype<OutT>();
    bool ret =
        bladnn::gemm(&bladnn_ctx, in_dtype, tp_a, A.data, A.sizes[0],
                     A.sizes[1], in_dtype, tp_b, B.data, B.sizes[0], B.sizes[1],
                     out_dtype, C.data, C.sizes[0], C.sizes[1]);
    if (ret) {
      return;
    }
  }
#endif

  auto lhs_matrix =
      makeMatrixDescriptor(ctx, A.data, A.sizes[0], A.sizes[1], tp_a);
  auto rhs_matrix =
      makeMatrixDescriptor(ctx, B.data, B.sizes[0], B.sizes[1], tp_b);
  // output remain default layout
  auto output_matrix =
      makeMatrixDescriptor(ctx, C.data, C.sizes[0], C.sizes[1], false);

  TAO_VLOG(1) << "A.data = " << A.data;
  TAO_VLOG(1) << "B.data = " << B.data;
  TAO_VLOG(1) << "C.data = " << C.data;
  TAO_VLOG(1) << "tp_a = " << tp_a << ", tp_b = " << tp_b;
  if (TAO_VLOG_IS_ON(1)) {
    print_memref(A, "A");
    print_memref(B, "B");
    print_memref(C, "C");
  }

  absl::optional<se::blas::AlgorithmType> best_algo_wrapper = absl::nullopt;
  bool blaze_use_mlir = false;
#ifdef BLAZE_OPT
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("BLAZE_USE_TAO_MLIR", false,
                                             &blaze_use_mlir));
#endif

  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream =
      static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));

  bool disable_tune = true;
  tensorflow::ReadBoolFromEnvVar("TAO_DISABLE_CUDA_GEMM_TUNE", true,
                                 &disable_tune);
  if (!blaze_use_mlir && !disable_tune) {
    se::blas::AlgorithmType best_algo;
    GemmTuningCacheKey key = makeGemmTuningCacheKey(
        A.sizes[0], A.sizes[1], tp_a, B.sizes[0], B.sizes[1], tp_b);
    std::string unique_name =
        "tao_ral.gpu.gemm_" + tao::ral::TaoTypeNameHelper<InT>::Invoke();
    auto state = ctx->getOrCreateResource<RalGemmState>(
        unique_name, []() { return new RalGemmState; });
    {
      std::lock_guard<std::mutex> l(state->mu);
      auto it = state->gemm_tuning_cache.find(key);
      if (it == state->gemm_tuning_cache.end()) {
        best_algo = tuningGemm<InT, OutT, E>(stream, lhs_matrix, rhs_matrix,
                                             output_matrix);
        state->gemm_tuning_cache.emplace(key, best_algo);
      } else {
        best_algo = it->second;
      }
      best_algo_wrapper = absl::make_optional(best_algo);
    }
  }

  auto s = DoGemmWithAlgorithm<InT, OutT, E>(
      /*batch_size*/ 1, lhs_matrix, rhs_matrix, output_matrix,
      /*alpha*/ E(1.),
      /*beta*/ E(0.), stream, best_algo_wrapper,
      /*output_profile_result=*/nullptr);

  if (!s) {
    TAO_VLOG(0) << "gemm fails to launch";
    ctx->signalError(Context::FAILURE, "fail to launch gemm");
  }
}

template <typename T, int N>
int64_t GetBatchSize(MemRefType<T, N> memref) {
  int64_t batch = 1;
  for (int64_t i = 0; i < N - 2; ++i) {
    batch *= memref.sizes[i];
  }
  return batch;
}

template <typename InT, typename OutT, int N, typename E = float>
void ral_batch_gemm(ExecutionContext* ctx, void* stream_handle,
                    MemRefType<InT, N> A, MemRefType<InT, N> B,
                    MemRefType<OutT, N> C, bool tp_a, bool tp_b,
                    bool weight_is_const) {
  if (isEmptyMemref(A) || isEmptyMemref(B) || isEmptyMemref(C)) {
    TAO_VLOG(1) << "ral_batch_gemm: early return for empty tensor";
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

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  {
    auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
    auto stream =
        static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));
    void* s = stream->implementation()->GpuStreamHack();
    bladnn::Context bladnn_ctx{s};
    bladnn::Dtype in_dtype = toBlaDNNDtype<InT>();
    bladnn::Dtype out_dtype = toBlaDNNDtype<OutT>();
    bool ret = bladnn::gemm(&bladnn_ctx, in_dtype, tp_a, A.data, A.sizes[N - 2],
                            A.sizes[N - 1], in_dtype, tp_b, B.data,
                            B.sizes[N - 2], B.sizes[N - 1], out_dtype, C.data,
                            C.sizes[N - 2], C.sizes[N - 1], batch_a);
    if (ret) {
      return;
    }
  }
#endif

  auto lhs_matrix = makeMatrixDescriptor(ctx, A.data, A.sizes[N - 2],
                                         A.sizes[N - 1], tp_a, batch_a);
  auto rhs_matrix = makeMatrixDescriptor(ctx, B.data, B.sizes[N - 2],
                                         B.sizes[N - 1], tp_b, batch_b);
  // output remain default layout
  auto output_matrix = makeMatrixDescriptor(ctx, C.data, C.sizes[N - 2],
                                            C.sizes[N - 1], false, batch_c);

  TAO_VLOG(1) << "A.data = " << A.data;
  TAO_VLOG(1) << "B.data = " << B.data;
  TAO_VLOG(1) << "C.data = " << C.data;
  TAO_VLOG(1) << "tp_a = " << tp_a << ", tp_b = " << tp_b;
  TAO_VLOG(1) << "batch size = " << batch_a;
  if (TAO_VLOG_IS_ON(1)) {
    print_memref(A, "A");
    print_memref(B, "B");
    print_memref(C, "C");
  }

  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream =
      static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));

  // Batch gemm does not support tuning ATM.
  absl::optional<se::blas::AlgorithmType> algo;
  auto s = DoGemmWithAlgorithm<InT, OutT, E>(
      /*batch_size*/ batch_a, lhs_matrix, rhs_matrix, output_matrix,
      /*alpha*/ 1.,
      /*beta*/ 0., stream, algo,
      /*output_profile_result=*/nullptr);

  if (!s) {
    ctx->signalError(Context::FAILURE, "fail to launch gemm");
  }
}

////////////////////////////////////////////////////////////////////////
///////////////           GpuGemmImpl Finish
///////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
///////////////           GpuConvImpl Begin
///////////////
////////////////////////////////////////////////////////////////////////
namespace gpu_conv_impl {

template <typename T, int N>
se::DeviceMemoryBase GetDeviceAddress(MemRefType<T, N> in) {
  int64_t bytes = sizeof(T);
  for (int i = 0; i < N; ++i) {
    bytes *= in.sizes[i];
  }
  return se::DeviceMemoryBase(reinterpret_cast<char*>(in.data), bytes);
}

using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::Stream;
using se::dnn::AlgorithmConfig;
using se::dnn::AlgorithmDesc;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::DimIndex;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;
using se::dnn::ProfileResult;

#if (TF_MAJOR_VERSION > 1 || TF_MINOR_VERSION > 12) || TENSORFLOW_USE_ROCM
// rocm backend with host tensorflow<=1.12 is not supported for now
using se::dnn::ConvolutionKind;
#else
enum class ConvolutionKind {
  FORWARD,
  FORWARD_BIAS_ACTIVATION,
  BACKWARD_DATA,
  BACKWARD_FILTER
};
#endif

struct CudnnConvParams;
class ScratchAllocator;

struct CudnnConvParamsKey {
  std::vector<int64_t> input_shape;
  std::vector<int64_t> filter_shape;
  std::vector<int64_t> paddings;
  std::vector<int64_t> output_shape;
  // strides & dilation & layouts
  std::vector<int64_t> metadata;
};

struct CudnnConvParamsKeyHasher {
  std::size_t operator()(const CudnnConvParamsKey& k) const {
    auto h = std::hash<int>()(0);
    for (auto vec : {&k.input_shape, &k.filter_shape, &k.paddings,
                     &k.output_shape, &k.metadata}) {
      h = std::hash<size_t>()(vec->size()) ^ (h << 1);
      for (auto v : *vec) {
        h = std::hash<size_t>()(v) ^ (h << 1);
      }
    }
    return h;
  }
};

struct CudnnConvParamsKeyEqual {
  bool operator()(const CudnnConvParamsKey& lhs,
                  const CudnnConvParamsKey& rhs) const {
    return (
        lhs.input_shape == rhs.input_shape &&
        lhs.filter_shape == rhs.filter_shape && lhs.paddings == rhs.paddings &&
        lhs.output_shape == rhs.output_shape && lhs.metadata == rhs.metadata);
  }
};

struct CudnnConvParams {
  ConvolutionKind kind;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> rhs_dilations;
  std::vector<std::pair<int64_t, int64_t>> paddings;

  std::vector<int64_t> input_shape;
  std::vector<int64_t> filter_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> metadata;

  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;

  int64_t algo_id;
  size_t best_result_bytes_used;
  bool tensor_ops_enabled;

  BatchDescriptor input_descriptor;
  FilterDescriptor filter_descriptor;
  ConvolutionDescriptor convolution_descriptor;
  BatchDescriptor output_descriptor;
};

#if TENSORFLOW_USE_ROCM

template <typename T>
std::vector<ProfileResult> GetMIOpenAlgorithms(
    ExecutionContext* ctx, CudnnConvParams& params,
    se::StreamExecutor* stream_exec, se::Stream* stream,
    std::vector<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer, ScratchAllocator* scratch_allocator) {
  std::vector<ProfileResult> algorithms;
  if (!stream_exec->GetMIOpenConvolveAlgorithms(
          params.kind, se::dnn::ToDataType<T>::value, stream,
          params.input_descriptor, operand_buffers[0], params.filter_descriptor,
          operand_buffers[1], params.output_descriptor, result_buffer,
          params.convolution_descriptor, scratch_allocator, &algorithms)) {
    ctx->signalError(Context::FAILURE, "GetMIOpenAlgorithms failed.");
  }
  return algorithms;
}

#else

std::vector<AlgorithmDesc> GetAlgorithms(ConvolutionKind kind,
                                         se::StreamExecutor* stream_exec) {
  std::vector<AlgorithmDesc> algorithms;
  bool succ = false;
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 6) || TF_MAJOR_VERSION > 2
  // TF2.7 and later
  succ = stream_exec->GetConvolveAlgorithms(kind, &algorithms);
#elif (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 5)
  // TF2.6
  switch (kind) {
    case ConvolutionKind::BACKWARD_FILTER:
      succ = stream_exec->GetConvolveBackwardFilterAlgorithms(&algorithms);
      break;
    case ConvolutionKind::BACKWARD_DATA:
      succ = stream_exec->GetConvolveBackwardDataAlgorithms(&algorithms);
      break;
    case ConvolutionKind::FORWARD:
    case ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      succ = stream_exec->GetConvolveAlgorithms(&algorithms);
      break;
  }
#else
  // TF2.4 TF1.12, TF1.15
  switch (kind) {
    case ConvolutionKind::BACKWARD_FILTER:
      succ =
          stream_exec->GetConvolveBackwardFilterAlgorithms(true, &algorithms);
      break;
    case ConvolutionKind::BACKWARD_DATA:
      succ = stream_exec->GetConvolveBackwardDataAlgorithms(true, &algorithms);
      break;
    case ConvolutionKind::FORWARD:
    case ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      succ = stream_exec->GetConvolveAlgorithms(true, &algorithms);
      break;
  }
#endif
  return algorithms;
}

#endif  // TENSORFLOW_USE_ROCM

// process-level states, using to store the conv tuning result.
// Not using a context-level state to reduce peak memory consumption.
struct RalConvState {
  static RalConvState& Get() {
    static RalConvState* state = new RalConvState;
    return *state;
  }
  // used to protect conv tuning process
  std::mutex tuning_mu;
  // cache mutex, used to protect cache_table
  std::mutex mu;
  using Cache =
      std::unordered_map<CudnnConvParamsKey, CudnnConvParams,
                         CudnnConvParamsKeyHasher, CudnnConvParamsKeyEqual>;
  std::unordered_map<std::string, Cache> cache_table;
};

template <typename T, int N>
CudnnConvParamsKey makeConvTuningCacheKey(MemRefType<T, N>& input,
                                          MemRefType<T, N>& filter,
                                          MemRefType<int32_t, 1>& paddings,
                                          MemRefType<T, N>& output,
                                          MemRefType<int32_t, 1>& metadata) {
  // Note metadata order:
  //   - input layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - kernel layout: each field for one dimension. The order is:
  //     * in_channel, out_channel, spatial dimensions
  //   - output layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - strides: each filed for one spatial dimension.
  //   - dilations: each filed for one spatial dimension.
  CudnnConvParamsKey key;
  static_assert(N > 2);
  const int effectiveN = (N < 4) ? 4 : N;
  bool is_1d_conv = (N == 3);
  std::vector<int64_t> input_sizes(effectiveN);
  std::vector<int64_t> filter_sizes(effectiveN);
  std::vector<int64_t> paddings_sizes(paddings.sizes[0] +
                                      static_cast<int64_t>(is_1d_conv) * 2);
  std::vector<int64_t> output_sizes(effectiveN);
  std::vector<int64_t> metadata_data(effectiveN * 3 + (effectiveN - 2) * 2);

  std::vector<std::vector<int64_t>*> target_sizes(
      {&input_sizes, &filter_sizes, &output_sizes});
  std::vector<int64_t*> source_sizes(
      {&(input.sizes[0]), &(filter.sizes[0]), &(output.sizes[0])});
  // input, kernel, and output layout & sizes.
  for (int64_t layout_idx = 0; layout_idx < 3; layout_idx++) {
    std::vector<int64_t>* target = target_sizes[layout_idx];
    int64_t* source = source_sizes[layout_idx];
    int64_t src_meta_idx_base = layout_idx * N;
    int64_t spatial_dim = metadata.data[src_meta_idx_base + 2];
    int64_t target_meta_idx_base = layout_idx * effectiveN;
    for (int64_t i = 0; i < N; i++) {
      int64_t src_meta_idx = src_meta_idx_base + i;
      int64_t src_dim = metadata.data[src_meta_idx];
      int64_t target_dim = src_dim;
      int64_t target_meta_idx = target_meta_idx_base + i;
      if (is_1d_conv) {
        // We will insert one spatial dim just before the existing spatial dim.
        // The dims after spatial dim, including the existing spatial dim, will
        // move right one step. The index of existing spatial dim in metadata
        // array will also move right.
        if (src_dim >= spatial_dim) {
          target_dim++;
        }
        if (i == N - 1) {
          target_meta_idx++;
        }
      }
      metadata_data[target_meta_idx] = target_dim;
      (*target)[target_dim] = source[src_dim];
    }
    if (is_1d_conv) {
      metadata_data[target_meta_idx_base + effectiveN - 2] = spatial_dim;
      (*target)[spatial_dim] = 1;
    }
  }
  // strides and dilations.
  for (int64_t idx = 0; idx < 2; idx++) {
    int64_t src_meta_idx_base = N * 3 + (N - 2) * idx;
    int64_t target_meta_idx_base = effectiveN * 3 + (effectiveN - 2) * idx;
    if (is_1d_conv) {
      // The new dim is insert before existing spatial dims for Conv1D.
      metadata_data[target_meta_idx_base++] = 1;
    }
    for (int64_t i = 0; i < N - 2; i++) {
      metadata_data[target_meta_idx_base++] =
          metadata.data[src_meta_idx_base + i];
    }
  }
  // paddings.
  if (is_1d_conv) {
    // Note that the new dim is insert before existing spatial dims for Conv1D.
    paddings_sizes[0] = 0;
    paddings_sizes[1] = 0;
  }
  int64_t offset = static_cast<int64>(is_1d_conv) * 2;
  for (int i = 0; i < paddings.sizes[0]; ++i) {
    paddings_sizes[offset + i] = paddings.data[i];
  }

  key.input_shape = std::move(input_sizes);
  key.filter_shape = std::move(filter_sizes);
  key.paddings = std::move(paddings_sizes);
  key.output_shape = std::move(output_sizes);
  key.metadata = std::move(metadata_data);

  return key;
}

std::unique_ptr<std::vector<int64_t>> StreamExecutorConvLayoutsToMetadata(
    DataLayout input, FilterLayout filter, DataLayout output) {
  std::unique_ptr<std::vector<int64_t>> layouts_ptr{new std::vector<int64_t>};
  auto& layouts = *layouts_ptr;
  switch (input) {
    case DataLayout::kBatchDepthYX:
      layouts.push_back(0);
      layouts.push_back(1);
      layouts.push_back(2);
      layouts.push_back(3);
      break;
    case DataLayout::kBatchYXDepth:
      layouts.push_back(0);
      layouts.push_back(3);
      layouts.push_back(1);
      layouts.push_back(2);
      break;
    default:
      return nullptr;
  }

  switch (filter) {
    case FilterLayout::kOutputInputYX:
      layouts.push_back(1);
      layouts.push_back(0);
      layouts.push_back(2);
      layouts.push_back(3);
      break;
    case FilterLayout::kOutputYXInput:
      layouts.push_back(3);
      layouts.push_back(0);
      layouts.push_back(1);
      layouts.push_back(2);
      break;
    default:
      return nullptr;
  }

  switch (output) {
    case DataLayout::kBatchDepthYX:
      layouts.push_back(0);
      layouts.push_back(1);
      layouts.push_back(2);
      layouts.push_back(3);
      break;
    case DataLayout::kBatchYXDepth:
      layouts.push_back(0);
      layouts.push_back(3);
      layouts.push_back(1);
      layouts.push_back(2);
      break;
    default:
      return nullptr;
  }

  return std::move(layouts_ptr);
}

struct Layouts {
  DataLayout input_dl;
  FilterLayout filter_dl;
  DataLayout output_dl;
  std::vector<int64_t> metadata;

  bool Match(const std::vector<int64_t>& metadata) {
    if (metadata.size() < this->metadata.size()) {
      return false;
    }
    for (size_t i = 0; i < this->metadata.size(); ++i) {
      if (metadata[i] != this->metadata[i]) {
        return false;
      }
    }
    return true;
  }
};

std::vector<Layouts> initSupportedLayouts() {
  std::vector<Layouts> layouts;
  // NCHW + OIHW
  {
    auto layouts_ptr = StreamExecutorConvLayoutsToMetadata(
        DataLayout::kBatchDepthYX, FilterLayout::kOutputInputYX,
        DataLayout::kBatchDepthYX);
    if (layouts_ptr) {
      layouts.emplace_back(
          Layouts{DataLayout::kBatchDepthYX, FilterLayout::kOutputInputYX,
                  DataLayout::kBatchDepthYX, std::move(*layouts_ptr)});
    }
  }

  // NCHW + OHWI
  {
    auto layouts_ptr = StreamExecutorConvLayoutsToMetadata(
        DataLayout::kBatchDepthYX, FilterLayout::kOutputYXInput,
        DataLayout::kBatchDepthYX);
    if (layouts_ptr) {
      layouts.emplace_back(
          Layouts{DataLayout::kBatchDepthYX, FilterLayout::kOutputYXInput,
                  DataLayout::kBatchDepthYX, std::move(*layouts_ptr)});
    }
  }

  // NHWC + OHWI
  {
    auto layouts_ptr = StreamExecutorConvLayoutsToMetadata(
        DataLayout::kBatchYXDepth, FilterLayout::kOutputYXInput,
        DataLayout::kBatchYXDepth);
    if (layouts_ptr) {
      layouts.emplace_back(
          Layouts{DataLayout::kBatchYXDepth, FilterLayout::kOutputYXInput,
                  DataLayout::kBatchYXDepth, std::move(*layouts_ptr)});
    }
  }

  return layouts;
}

Layouts* getLayout(const CudnnConvParamsKey& key) {
  static std::vector<Layouts> supported_layouts = initSupportedLayouts();

  Layouts* layouts = nullptr;
  for (auto& l : supported_layouts) {
    if (l.Match(key.metadata)) {
      layouts = &l;
      break;
    }
  }
  return layouts;
}

void FillStridesAndDilation(CudnnConvParams& params,
                            const CudnnConvParamsKey& key) {
  // Metadata:
  //   - input layput: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - kernel layout: each field for one dimension. The order is:
  //     * in_channel, out_channel, spatial dimensions
  //   - output layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - strides: each filed for one spatial dimension.
  //   - dilations: each filed for one spatial dimension.
  int rank = static_cast<int>(key.input_shape.size());

  // skip input layout, kernel layout and output layout
  int offset = 3 * rank;

  // strides
  for (int i = 0; i < rank - 2; ++i) {
    params.window_strides.push_back(key.metadata[offset++]);
  }

  // dilation
  for (int i = 0; i < rank - 2; ++i) {
    params.rhs_dilations.push_back(key.metadata[offset++]);
  }
}

void FillDescriptors(CudnnConvParams& params) {
  // rank of input/output/filter
  const int rank = static_cast<int>(params.input_shape.size());
  // # of spatial dimensions
  const int num_dimensions = static_cast<int>(params.window_strides.size());
  CHECK_LE(num_dimensions, 3);
  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  int offset = 0;
  auto& m = params.metadata;
  auto& input_shape = params.input_shape;
  auto& filter_shape = params.filter_shape;
  auto& output_shape = params.output_shape;
  auto& paddings = params.paddings;
  auto& window_strides = params.window_strides;
  auto& dilations = params.rhs_dilations;

  if (TAO_VLOG_IS_ON(2)) {
    TAO_VLOG(0) << "CudnnConvParams:\n"
                << "\tinput rank: " << rank << "\n"
                << "\tnum_dimensions: " << num_dimensions << "\n"
                << "\teffective_num_dimensions: " << effective_num_dimensions
                << "\n";
    TAO_VLOG(0) << "\tmetadata:\n";
    for (size_t i = 0; i < m.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << m[i];
    }
    TAO_VLOG(0) << "\tinput_shape:\n";
    for (size_t i = 0; i < input_shape.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << input_shape[i];
    }
    TAO_VLOG(0) << "\tfilter_shape:\n";
    for (size_t i = 0; i < filter_shape.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << filter_shape[i];
    }
    TAO_VLOG(0) << "\toutput_shape:\n";
    for (size_t i = 0; i < output_shape.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << output_shape[i];
    }
    TAO_VLOG(0) << "\tpaddings:\n";
    for (size_t i = 0; i < paddings.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << paddings[i].first << ", "
                  << paddings[i].second;
    }
    TAO_VLOG(0) << "\twindow_strides:\n";
    for (size_t i = 0; i < window_strides.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << window_strides[i];
    }
    TAO_VLOG(0) << "\tdilations:\n";
    for (size_t i = 0; i < dilations.size(); ++i) {
      TAO_VLOG(0) << "\t  #" << i << ": " << dilations[i];
    }
  }

  // input descriptor
  BatchDescriptor input_descriptor(effective_num_dimensions);
  input_descriptor.set_layout(params.input_dl);
  input_descriptor.set_count(input_shape[m[offset++]]);
  input_descriptor.set_feature_map_count(input_shape[m[offset++]]);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        params.input_shape[m[dim + offset]]);
  }
  offset += num_dimensions;
  params.input_descriptor.CloneFrom(input_descriptor);

  FilterDescriptor filter_descriptor(effective_num_dimensions);
  filter_descriptor.set_layout(params.filter_dl);
  filter_descriptor.set_input_feature_map_count(filter_shape[m[offset++]]);
  filter_descriptor.set_output_feature_map_count(filter_shape[m[offset++]]);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape[m[dim + offset]]);
  }
  offset += num_dimensions;
  params.filter_descriptor.CloneFrom(filter_descriptor);

  int64_t input_filter_count = input_shape[m[1]];
  int64_t filter_input_feature_map_count = filter_shape[m[rank]];
  int64_t feature_group_count =
      input_filter_count / filter_input_feature_map_count;
  ConvolutionDescriptor convolution_descriptor(num_dimensions);
  convolution_descriptor.set_group_count(feature_group_count);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    convolution_descriptor
        .set_zero_padding(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            paddings[dim].first)
        .set_dilation_rate(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            dilations[dim])
        .set_filter_stride(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window_strides[dim]);
  }
  params.convolution_descriptor = convolution_descriptor;

  BatchDescriptor output_descriptor(effective_num_dimensions);
  output_descriptor.set_layout(params.output_dl);
  output_descriptor.set_count(output_shape[m[offset++]]);
  output_descriptor.set_feature_map_count(output_shape[m[offset++]]);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        output_shape[m[dim + offset]]);
  }
  offset += num_dimensions;
  params.output_descriptor.CloneFrom(output_descriptor);

  if (TAO_VLOG_IS_ON(2)) {
    TAO_VLOG(0) << "\tdescriptors:\n"
                << "\t  input: " << params.input_descriptor.ToString() << "\n"
                << "\t  filter: " << params.filter_descriptor.ToString() << "\n"
                << "\t  conv: " << params.convolution_descriptor.ToString()
                << "\n"
                << "\t  output: " << params.output_descriptor.ToString()
                << "\n";
  }
}

std::unique_ptr<CudnnConvParams> makeNewConvParams(
    const CudnnConvParamsKey& key) {
  std::unique_ptr<CudnnConvParams> params_ptr(new CudnnConvParams);
  auto& params = *params_ptr;

  auto layouts = getLayout(key);
  if (!layouts) {
    TAO_VLOG(0) << "unsupported conv layout";
    return nullptr;
  }

  params.kind = ConvolutionKind::FORWARD;
  params.input_shape = key.input_shape;
  params.filter_shape = key.filter_shape;
  params.output_shape = key.output_shape;
  params.metadata = layouts->metadata;
  params.input_dl = layouts->input_dl;
  params.filter_dl = layouts->filter_dl;
  params.output_dl = layouts->output_dl;

  int rank = static_cast<int>(key.input_shape.size());
  for (int i = 0; i < rank - 2; ++i) {
    // padding_low & padding_high for each spatial dimension
    params.paddings.emplace_back(
        std::make_pair(key.paddings[2 * i], key.paddings[2 * i + 1]));
  }
  FillStridesAndDilation(params, key);
  FillDescriptors(params);
  return std::move(params_ptr);
}

struct ScopedBuffer {
  ScopedBuffer() = default;
  ScopedBuffer(ExecutionContext* ctx, GPUDriver* gpu_driver, void* ptr)
      : ctx_(ctx), gpu_driver_(gpu_driver), ptr_(ptr) {}
  ~ScopedBuffer() { gpu_driver_->dealloc(ctx_, ptr_); }
  ExecutionContext* ctx_;
  GPUDriver* gpu_driver_;
  void* ptr_;
};

class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(ExecutionContext* ctx, GPUDriver* gpu_driver)
      : ctx_(ctx), gpu_driver_(gpu_driver) {}

#if defined(TF_1_12) || defined(TF_1_14)
  int64 GetMemoryLimitInBytes(se::Stream* stream) override {
    return GetMemoryLimitInBytesImpl();
  }

  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      se::Stream* stream, int64 byte_size) override {
    return AllocateBytesImpl(byte_size);
  }
#else
  int64 GetMemoryLimitInBytes() override { return GetMemoryLimitInBytesImpl(); }

  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64 byte_size) override {
    return AllocateBytesImpl(byte_size);
  }
#endif
  int64 TotalAllocatedBytes() { return total_allocated_bytes_; }

 private:
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytesImpl(
      int64 byte_size);
  // BFCAllocator is not exposed for the decoupled compiler.
  // Thus we don't have a "try allocate" mechanism in TaoBridge as in TF,
  // the host will crash once the amount of scratch memory tried to be
  // allocated exceeds the memories left.
  // TODO: For now we just set a small threshold to ease this problem.
  // Revisit this for the performance degrade in more models.
  int64 GetMemoryLimitInBytesImpl() {
    return 1LL << 28;  // 256M.
  }

 private:
  ExecutionContext* ctx_;
  GPUDriver* gpu_driver_;
  std::vector<std::unique_ptr<ScopedBuffer>> allocated_buffers_;
  int64 total_allocated_bytes_ = 0;
};

se::port::StatusOr<se::DeviceMemory<uint8>> ScratchAllocator::AllocateBytesImpl(
    int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytesImpl()) {
    return errors::Internal("Allocating buffer exceeds the memory limit");
  }

  TAO_VLOG(2) << "AllocateBytesImpl bytes: " << byte_size;
  void* ptr = gpu_driver_->alloc(ctx_, byte_size);

  if (!ptr) {
    TAO_VLOG(2) << "AllocateBytesImpl failed: OOM with bytes = " << byte_size;
    return errors::Internal("Allocating failed");
  }

  total_allocated_bytes_ += byte_size;
  se::DeviceMemoryBase buffer_addr(ptr, byte_size);
  allocated_buffers_.emplace_back(new ScopedBuffer(ctx_, gpu_driver_, ptr));
  return se::DeviceMemory<uint8>(buffer_addr);
}

template <typename T>
Status RunCudnnConvolution(CudnnConvParams& params,
                           std::vector<se::DeviceMemoryBase>& operand_buffers,
                           se::DeviceMemoryBase& result_buffer,
                           se::ScratchAllocator* scratch_allocator,
                           se::Stream* stream,
                           se::dnn::ProfileResult* profile_result) {
  ConvolutionKind kind = params.kind;
  DeviceMemory<T> input_buf(operand_buffers[0]);
  DeviceMemory<T> filter_buf(operand_buffers[1]);
  DeviceMemory<T> output_buf(result_buffer);
  auto& input_descriptor = params.input_descriptor;
  auto& filter_descriptor = params.filter_descriptor;
  auto& convolution_descriptor = params.convolution_descriptor;
  auto& output_descriptor = params.output_descriptor;
  AlgorithmConfig algorithm{
      AlgorithmDesc(params.algo_id, params.tensor_ops_enabled)};
#if TENSORFLOW_USE_ROCM
  if (profile_result) {
    algorithm.set_scratch_size(profile_result->scratch_size());
  } else {
    algorithm.set_scratch_size(params.best_result_bytes_used);
  }
#endif

  if (TAO_VLOG_IS_ON(2)) {
    TAO_VLOG(0) << "input ptr: " << input_buf.opaque() << "@"
                << input_buf.size();
    TAO_VLOG(0) << "filter_buf ptr: " << filter_buf.opaque() << "@"
                << filter_buf.size();
    TAO_VLOG(0) << "output_buf ptr: " << output_buf.opaque() << "@"
                << output_buf.size();
    TAO_VLOG(0) << "\tdescriptors:\n"
                << "\t  input: " << params.input_descriptor.ToString() << "\n"
                << "\t  filter: " << params.filter_descriptor.ToString() << "\n"
                << "\t  conv: " << params.convolution_descriptor.ToString()
                << "\n"
                << "\t  output: " << params.output_descriptor.ToString()
                << "\n";
    TAO_VLOG(0) << "\talgorithm: " << algorithm.ToString();
    TAO_VLOG(0) << "\tprofile_result: " << profile_result;
    TAO_VLOG(0) << "\tscratch: " << scratch_allocator;
  }

  Status status = Status::OK();
  switch (kind) {
    case ConvolutionKind::FORWARD:
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 6) || TF_MAJOR_VERSION > 2
      // TF2.7 and later
      status = stream->ConvolveWithAlgorithm(
          kind, input_descriptor, input_buf, filter_descriptor, filter_buf,
          output_descriptor, output_buf, convolution_descriptor,
          scratch_allocator, algorithm, profile_result);
#elif TF_MAJOR_VERSION > 1
      // TF2.4
      status = stream->ConvolveWithAlgorithm(
          input_descriptor, input_buf, filter_descriptor, filter_buf,
          convolution_descriptor, output_descriptor, &output_buf,
          scratch_allocator, algorithm, profile_result);
#else
      // TF1.12, TF1.15
      // Not return status in this kind of API, check the status of stream
      // instead.
      stream->ThenConvolveWithAlgorithm(
          input_descriptor, input_buf, filter_descriptor, filter_buf,
          convolution_descriptor, output_descriptor, &output_buf,
          scratch_allocator, algorithm, profile_result);
#endif
      break;
    default:
      return errors::Internal("Not known CudnnConvKind");
  }

  if (!status.ok()) return status;
  if (!stream->ok()) {
    return errors::Internal("Unable to launch convolution");
  }
  return Status::OK();
}

template <typename T>
bool PickBestAlgorithm(CudnnConvParams& params,
                       std::vector<se::DeviceMemoryBase>& operand_buffers,
                       se::DeviceMemoryBase result_buffer, se::Stream* stream,
                       ExecutionContext* ctx, GPUDriver* gpu_driver) {
  auto stream_exec = stream->parent();

  // exclusive tuning.
  std::lock_guard<std::mutex> l(RalConvState::Get().tuning_mu);

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec->SynchronizeAllActivity()) {
    ctx->signalError(Context::FAILURE,
                     "Failed to synchronize GPU for autotuning.");
    return false;
  }

  se::dnn::ProfileResult best_result;
  size_t best_result_bytes_used = 0;
  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm sufficies. It doesn't make
  // this algorithm considered correct, though.
  ScratchAllocator scratch_allocator(ctx, gpu_driver);
#if TENSORFLOW_USE_ROCM
  for (se::dnn::ProfileResult& profile_result :
       GetMIOpenAlgorithms<T>(ctx, params, stream_exec, stream, operand_buffers,
                              result_buffer, &scratch_allocator)) {
    params.algo_id = profile_result.algorithm().algo_id();
    params.tensor_ops_enabled = profile_result.algorithm().tensor_ops_enabled();
#else
  for (const AlgorithmDesc& alg : GetAlgorithms(params.kind, stream_exec)) {
    params.algo_id = alg.algo_id();
    params.tensor_ops_enabled = alg.tensor_ops_enabled();
    se::dnn::ProfileResult profile_result;
#endif
    Status launch_status =
        RunCudnnConvolution<T>(params, operand_buffers, result_buffer,
                               &scratch_allocator, stream, &profile_result);
    if (launch_status.ok() && profile_result.is_valid()) {
      int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
      TAO_VLOG(2) << "Run of conv algorithm succ: scratch_bytes_used = "
                  << scratch_bytes_used << ", elapsed_time_in_ms = "
                  << profile_result.elapsed_time_in_ms();
      if (profile_result.elapsed_time_in_ms() <
          best_result.elapsed_time_in_ms()) {
        best_result = profile_result;
        best_result_bytes_used = scratch_bytes_used;
      }
    } else {
      TAO_VLOG(2) << "Run of conv algorithm failed: "
                  << " profile_result valid = " << profile_result.is_valid()
                  << ", err_msg = " << launch_status.error_message();
    }
  }
  if (best_result.is_valid()) {
    params.algo_id = best_result.algorithm().algo_id();
    params.tensor_ops_enabled = best_result.algorithm().tensor_ops_enabled();
    params.best_result_bytes_used = best_result_bytes_used;
  }

  return true;
}  // namespace gpu_conv_impl

static bool layout_match(const std::vector<int32_t>& ref,
                         const MemRefType<int32_t, 1>& metadata) {
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
  static_assert(N > 2, "dimension should be larger than 2");
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

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  // nchw=0123 iohw=0123
  // const std::vector<int32_t> nchw_oihw_layout = {0, 1, 2, 3, 1, 0,
  //                                                2, 3, 0, 1, 2, 3};
  // const std::vector<int32_t> nhwc_hwio_layout = {0, 3, 1, 2, 2, 3,
  //                                                0, 1, 0, 3, 1, 2};
  const std::vector<int32_t> nhwc_ohwi_layout = {0, 3, 1, 2, 3, 0,
                                                 1, 2, 0, 3, 1, 2};
  if (layout_match(nhwc_ohwi_layout, metadata)) {
    auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
    auto stream =
        static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));
    void* s = stream->implementation()->GpuStreamHack();
    int32_t n = input.sizes[0];
    assert(n == output.sizes[0]);
    T* a_data = input.data;
    T* b_data = kernel.data;
    T* c_data = output.data;
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
    oh = output.sizes[1];
    ow = output.sizes[2];
    oc == output.sizes[3];
    assert(ko == oc);
    int pad_h = padding.data[0];
    int pad_w = padding.data[2];
    size_t offset = N * 3;
    int stride_h = metadata.data[offset++];
    int stride_w = metadata.data[offset++];
    int dilation_h = metadata.data[offset++];
    int dilation_w = metadata.data[offset++];
    bool is_depthwise = false;
    int32_t groups = 1;
    if (ic != ki) {
      assert(ki == 1);
      is_depthwise = true;
      groups = ic;
    }
    auto conv_kind = bladnn::ConvKind::kFprop;
    auto data_layout = bladnn::Layout::kNHWC;
    auto kernel_layout = bladnn::Layout::kNHWC;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    bladnn::Dtype dtype = toBlaDNNDtype<T>();
    bool ret = false;
    ret = bladnn::conv2d(s, dtype, dtype, conv_kind, data_layout, kernel_layout,
                         n, ih, iw, ic, ko, kh, kw, oh, ow, pad_h, pad_w,
                         stride_h, stride_w, dilation_h, dilation_w, groups,
                         &alpha, a_data, b_data, &beta, c_data, c_data);
    if (ret) {
      return;
    }
  }
#endif

  auto key = makeConvTuningCacheKey(input, kernel, padding, output, metadata);
  CudnnConvParams* params = nullptr;
  std::string unique_name =
      "tao_ral.gpu.conv_" + tao::ral::TaoTypeNameHelper<T>::Invoke();
  auto& state = RalConvState::Get();
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream =
      static_cast<se::Stream*>(gpu_driver->asSEStream(ctx, stream_handle));

  std::vector<se::DeviceMemoryBase> operand_se_buffers;
  operand_se_buffers.emplace_back(GetDeviceAddress(input));
  operand_se_buffers.emplace_back(GetDeviceAddress(kernel));
  se::DeviceMemoryBase result_buffer = GetDeviceAddress(output);

  {
    std::lock_guard<std::mutex> l(state.mu);
    auto& cache = state.cache_table[unique_name];
    auto it = cache.find(key);
    if (it == cache.end()) {
      auto params_ptr = makeNewConvParams(key);
      if (!params_ptr) {
        ctx->signalError(Context::FAILURE, "illegal conv op");
        return;
      }

      if (!PickBestAlgorithm<T>(*params_ptr, operand_se_buffers, result_buffer,
                                stream, ctx, gpu_driver)) {
        ctx->signalError(Context::FAILURE, "fail to tune conv op");
        return;
      }

      it = cache.emplace(key, std::move(*params_ptr)).first;
    }
    params = &it->second;
  }

  if (params == nullptr) {
    ctx->signalError(Context::FAILURE,
                     "unable to build conv params to launch a conv");
    return;
  }

  ScratchAllocator scratch_allocator(ctx, gpu_driver);
  Status launch_status = RunCudnnConvolution<T>(
      *params, operand_se_buffers, result_buffer, &scratch_allocator, stream,
      /*profile_result*/ nullptr);
  if (!launch_status.ok()) {
    std::string err_msg =
        "d_conv launch failed: " + launch_status.error_message();
    ctx->signalError(Context::FAILURE, err_msg);
  }
  return;
}

}  // namespace gpu_conv_impl

////////////////////////////////////////////////////////////////////////
///////////////           GpuConvImpl Finish
///////////////
////////////////////////////////////////////////////////////////////////

}  // namespace se_impl
}  // namespace gpu

// gemm ops
TAO_RAL_API("ral_gemm", "gpu", gpu::se_impl::ral_gemm<float, float>);
TAO_RAL_API("ral_gemm", "gpu", gpu::se_impl::ral_gemm<double, double, double>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_gemm<Eigen::half, Eigen::half>);
TAO_RAL_API("ral_gemm", "gpu", gpu::se_impl::ral_batch_gemm<float, float, 3>);
TAO_RAL_API("ral_gemm", "gpu", gpu::se_impl::ral_batch_gemm<float, float, 4>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<double, double, 3, double>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<double, double, 4, double>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<Eigen::half, Eigen::half, 3>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<Eigen::half, Eigen::half, 4>);
#ifdef BLAZE_OPT
TAO_RAL_API("ral_gemm", "gpu", gpu::se_impl::ral_gemm<Eigen::half, float>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<Eigen::half, float, 3>);
TAO_RAL_API("ral_gemm", "gpu",
            gpu::se_impl::ral_batch_gemm<Eigen::half, float, 4>);
#endif
// conv ops
TAO_RAL_API("ral_conv", "gpu", gpu::se_impl::gpu_conv_impl::ral_conv<float, 4>);
TAO_RAL_API("ral_conv", "gpu",
            gpu::se_impl::gpu_conv_impl::ral_conv<Eigen::half, 4>);
TAO_RAL_API("ral_conv", "gpu", gpu::se_impl::gpu_conv_impl::ral_conv<float, 3>);
TAO_RAL_API("ral_conv", "gpu",
            gpu::se_impl::gpu_conv_impl::ral_conv<Eigen::half, 3>);

}  // namespace ral
}  // namespace tao

#endif
