#include "bladnn/backend/tvm/tvm_collector.h"
#include "bladnn/backend/tvm/tvm_handle.h"
#include "bladnn/bladnn.h"
#include "bladnn/utils/log.h"
#include "hip/hip_runtime.h"
#include "rotlass_handle.h"

namespace bladnn {
struct CutlassGemmTuningCacheKey {
  CutlassGemmTuningCacheKey(Dtype a_dtype, Dtype c_dtype, int m, int n, int k,
                            bool lhs_transpose, bool rhs_transpose,
                            int batch_count)
      : a_dtype(a_dtype),
        c_dtype(c_dtype),
        m(m),
        n(n),
        k(k),
        batch_count(batch_count),
        lhs_transpose(lhs_transpose),
        rhs_transpose(rhs_transpose) {}
  Dtype a_dtype;
  Dtype c_dtype;
  int m;
  int n;
  int k;
  bool lhs_transpose;
  bool rhs_transpose;
  int batch_count;
  bool operator<(const CutlassGemmTuningCacheKey& other) const {
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
    } else if (a_dtype != other.a_dtype) {
      return (a_dtype < other.a_dtype);
    } else if (c_dtype != other.c_dtype) {
      return (c_dtype < other.c_dtype);
    } else if (batch_count != other.batch_count) {
      return (batch_count < other.batch_count);
    } else {
      return false;
    }
  }
};

static bool enable_jit_ =
    cutlass::library::ReadBoolFromEnvVar("BLADE_GEMM_TUNE_JIT");
static bool verbose_ =
    cutlass::library::ReadBoolFromEnvVar("BLADE_GEMM_VERBOSE");
using namespace bladnn::tvm;

bool rotlass_gemm(Context* ctx, Dtype a_dtype, bool a_transpose,
                  const void* a_ptr, int a_dim0, int a_dim1, Dtype b_dtype,
                  bool b_transpose, const void* b_ptr, int b_dim0, int b_dim1,
                  Dtype c_dtype, void* c_ptr, int c_dim0, int c_dim1,
                  int batch_count, bool a_is_const, bool b_is_const,
                  const void* alpha, const void* beta) {
  if (!enable_jit_) {
    return false;
  }
  if (a_dtype != b_dtype ||
      (a_dtype != Dtype::kF64 && a_dtype != Dtype::kF32) ||
      (c_dtype != Dtype::kF64 && c_dtype != Dtype::kF32)) {
    return false;
  }
  if (batch_count < 1) {
    return false;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(ctx->stream);
  auto major_a = a_transpose ? cutlass::library::LayoutTypeID::kRowMajor
                             : cutlass::library::LayoutTypeID::kColumnMajor;
  auto major_b = b_transpose ? cutlass::library::LayoutTypeID::kRowMajor
                             : cutlass::library::LayoutTypeID::kColumnMajor;
  auto in_type = a_dtype == Dtype::kF64 ? cutlass::library::NumericTypeID::kF64
                                        : cutlass::library::NumericTypeID::kF32;
  auto out_type = c_dtype == Dtype::kF64
                      ? cutlass::library::NumericTypeID::kF64
                      : cutlass::library::NumericTypeID::kF32;
  auto acc_type = c_dtype == Dtype::kF64
                      ? cutlass::library::NumericTypeID::kF64
                      : cutlass::library::NumericTypeID::kF32;
  auto scale_type = c_dtype == Dtype::kF64
                        ? cutlass::library::NumericTypeID::kF64
                        : cutlass::library::NumericTypeID::kF32;
  static std::map<CutlassGemmTuningCacheKey, cutlass::library::TuneConfig>
      configs;
  static cutlass::library::BladeHandle handle;
  static std::mutex mu_;

  int m = a_transpose ? a_dim1 : a_dim0;
  int k = a_transpose ? a_dim0 : a_dim1;
  int n = b_transpose ? b_dim0 : b_dim1;
  int lda = a_dim1;
  int ldb = b_dim1;
  int ldc = c_dim1;
  const double dalpha_val = 1.0;
  const float falpha_val = 1.0f;
  if (alpha == nullptr) {
    alpha = &dalpha_val;
    if (c_dtype != Dtype::kF64) {
      alpha = &falpha_val;
    }
  }

  const double dbeta_val = 0.0;
  const float fbeta_val = 0.0f;
  if (beta == nullptr) {
    beta = &dbeta_val;
    if (c_dtype != Dtype::kF64) {
      beta = &fbeta_val;
    }
  }
  CutlassGemmTuningCacheKey key(a_dtype, c_dtype, m, n, k, a_transpose,
                                b_transpose, batch_count);
  handle.set_stream(stream);
  auto mode = batch_count > 1 ? cutlass::library::GemmUniversalMode::kBatched
                              : cutlass::library::GemmUniversalMode::kGemm;
  auto iter = configs.find(key);
  if (iter == configs.end()) {
    std::lock_guard<std::mutex> l(mu_);
    cutlass::library::TuneConfig tune_config;
    cutlass::Status status =
        handle.gemm_tune(mode, n, m, k, acc_type, scale_type,
                         alpha,    // pointer to alpha scalar
                         in_type,  // data type of A matrix
                         major_b,  // layout of A matrix
                         cutlass::library::ComplexTransform::kNone,
                         b_ptr,    // pointer to A matrix in device memory
                         ldb,      // leading dimension of A matrix
                         in_type,  // data type of B matrix
                         major_a,  // layout of B matrix
                         cutlass::library::ComplexTransform::kNone,
                         a_ptr,     // pointer to B matrix in device memory
                         lda,       // leading dimension of B matrix
                         beta,      // pointer to beta scalar
                         out_type,  // data type of C and D matrix
                         c_ptr,     // pointer to C matrix in device memory
                         ldc,       // leading dimension fo C matrix
                         c_ptr,     // pointer to D matrix in device memory
                         ldc,       // leading dimension of D matrix
                         batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1,
                         c_dim0 * c_dim1, c_dim0 * c_dim1, &tune_config, true);
    if (status != cutlass::Status::kSuccess) {
      tune_config.valid = false;
    }
    if (verbose_) {
      Sig sig = tune_config.sig;
      BLADNN_VLOG(0) << "tune_config " << tune_config.valid << " " << sig.BX
                     << "," << sig.BY << "," << sig.BK << " " << sig.WX << ","
                     << sig.WY << "," << sig.WK << " " << sig.IX << ","
                     << sig.IY << "," << sig.IK << " " << sig.S << " "
                     << tune_config.split_k;
    }
    iter = configs.insert(std::make_pair(key, tune_config)).first;
  }

  if (iter == configs.end() || iter->second.valid != true) {
    // fallback to cublas
    // handle.gemm_cublas(n, m, k, &alpha, in_type, major_b, b_ptr, ldb,
    // in_type,
    //                    major_a, a_ptr, lda, &beta, out_type, c_ptr, ldc);
    return false;
  } else {
    auto tune_config = iter->second;
    cutlass::Status status =
        handle.gemm_tune(mode, n, m, k, acc_type, scale_type,
                         alpha,    // pointer to alpha scalar
                         in_type,  // data type of A matrix
                         major_b,  // layout of A matrix
                         cutlass::library::ComplexTransform::kNone,
                         b_ptr,    // pointer to A matrix in device memory
                         ldb,      // leading dimension of A matrix
                         in_type,  // data type of B matrix
                         major_a,  // layout of B matrix
                         cutlass::library::ComplexTransform::kNone,
                         a_ptr,     // pointer to B matrix in device memory
                         lda,       // leading dimension of B matrix
                         beta,      // pointer to beta scalar
                         out_type,  // data type of C and D matrix
                         c_ptr,     // pointer to C matrix in device memory
                         ldc,       // leading dimension fo C matrix
                         c_ptr,     // pointer to D matrix in device memory
                         ldc,       // leading dimension of D matrix
                         batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1,
                         c_dim0 * c_dim1, c_dim0 * c_dim1, &tune_config, false);
    return true;
  }

  return false;
}

bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, void* c_ptr,
          int c_dim0, int c_dim1, int batch_count, bool a_is_const,
          bool b_is_const, const void* alpha, const void* beta) {
  bool ret =
      rotlass_gemm(ctx, a_dtype, a_transpose, a_ptr, a_dim0, a_dim1, b_dtype,
                   b_transpose, b_ptr, b_dim0, b_dim1, c_dtype, c_ptr, c_dim0,
                   c_dim1, batch_count, a_is_const, b_is_const, alpha, beta);
  if (ret) {
    return ret;
  }
  TVMHandler* tvm_handler = TVMHandlerCreateOrGet();
  TVMFuncCache* tvm_impl_cache =
      tvm_handler->TVMFuncCacheCreateOrGet("gemm", tvm_handler->GPUDevice());
  auto lhs_transpose =
      a_transpose ? TVMGemmTranspose::Transpose : TVMGemmTranspose::NoTranspose;
  auto rhs_transpose =
      b_transpose ? TVMGemmTranspose::Transpose : TVMGemmTranspose::NoTranspose;
  auto m = a_transpose ? a_dim1 : a_dim0;
  auto k = b_transpose ? b_dim1 : b_dim0;
  auto n = b_transpose ? b_dim0 : b_dim1;
  tvm_handler->CheckStatus();
  std::string key;
  if (a_dtype == Dtype::kF32 && b_dtype == Dtype::kF32 &&
      c_dtype == Dtype::kF32) {
    key = TVMHandler::GetGemmTVMFuncKey<Dtype::kF32, Dtype::kF32, Dtype::kF32>(
        tvm_handler->GPUDevice(), m, n, k, lhs_transpose, rhs_transpose);
  } else if (a_dtype == Dtype::kF64 && b_dtype == Dtype::kF64 &&
             c_dtype == Dtype::kF64) {
    key = TVMHandler::GetGemmTVMFuncKey<Dtype::kF64, Dtype::kF64, Dtype::kF64>(
        tvm_handler->GPUDevice(), m, n, k, lhs_transpose, rhs_transpose);
  } else {
    return false;
  }
  tvm_handler->CollectorAddKernel(key);
  const auto& tvm_impl = tvm_impl_cache->LookUp(key);
  if (tvm_impl.IsHit()) {
    BLADNN_VLOG(1) << "Look up optimized func cache hit for " << key;
    std::vector<void*> args = {const_cast<void*>(a_ptr),
                               const_cast<void*>(b_ptr), c_ptr};

    return tvm_impl.Launch(ctx->stream, static_cast<void*>(args.data()),
                           sizeof(args));
  } else {
    BLADNN_VLOG(1) << "Optimized func cache miss for " << key;
  }
  return false;
}

}  // namespace bladnn
