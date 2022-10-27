#include <cublas_v2.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "bladnn/backend/cutlass/cutlass_handle.h"
#include "bladnn/bladnn.h"
#include "bladnn/utils/log.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/util.h"

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

struct CutlassSparseGemmTuningCacheKey {
  CutlassSparseGemmTuningCacheKey(Dtype a_dtype, Dtype c_dtype, Dtype e_dtype,
                                  int m, int n, int k, bool lhs_transpose,
                                  bool rhs_transpose, int batch_count)
      : a_dtype(a_dtype),
        c_dtype(c_dtype),
        e_dtype(e_dtype),
        m(m),
        n(n),
        k(k),
        batch_count(batch_count),
        lhs_transpose(lhs_transpose),
        rhs_transpose(rhs_transpose) {}
  Dtype a_dtype;
  Dtype c_dtype;
  Dtype e_dtype;
  int m;
  int n;
  int k;
  bool lhs_transpose;
  bool rhs_transpose;
  int batch_count;
  bool operator<(const CutlassSparseGemmTuningCacheKey& other) const {
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
    } else if (e_dtype != other.e_dtype) {
      return (e_dtype < other.e_dtype);
    } else if (batch_count != other.batch_count) {
      return (batch_count < other.batch_count);
    } else {
      return false;
    }
  }
};

struct CutlassConv2DTuningCacheKey {
  Dtype in_dtype;
  Dtype out_dtype;
  ConvKind conv_kind;
  Layout data_layout;
  Layout kernel_layout;
  int N;
  int H;
  int W;
  int C;
  int K;
  int R;
  int S;
  int P;
  int Q;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;

  bool operator==(const CutlassConv2DTuningCacheKey& rhs) const {
    return in_dtype == rhs.in_dtype && out_dtype == rhs.out_dtype &&
           conv_kind == rhs.conv_kind && data_layout == rhs.data_layout &&
           kernel_layout == rhs.kernel_layout && N == rhs.N && H == rhs.H &&
           W == rhs.W && C == rhs.C && K == rhs.K && R == rhs.R && S == rhs.S &&
           P == rhs.P && Q == rhs.Q && pad_h == rhs.pad_h &&
           pad_w == rhs.pad_w && stride_h == rhs.stride_h &&
           stride_w == rhs.stride_w && dilation_h == rhs.dilation_h &&
           dilation_w == rhs.dilation_w && groups == rhs.groups;
  }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
struct CutlassConv2DTuningCacheKeyHasher {
  std::size_t operator()(const CutlassConv2DTuningCacheKey& key) const {
    std::size_t seed = std::hash<Dtype>()(key.in_dtype);
    hash_combine(seed, key.out_dtype);
    hash_combine(seed, key.conv_kind);
    hash_combine(seed, key.data_layout);
    hash_combine(seed, key.kernel_layout);
    hash_combine(seed, key.N);
    hash_combine(seed, key.H);
    hash_combine(seed, key.W);
    hash_combine(seed, key.C);
    hash_combine(seed, key.K);
    hash_combine(seed, key.R);
    hash_combine(seed, key.S);
    hash_combine(seed, key.P);
    hash_combine(seed, key.Q);
    hash_combine(seed, key.pad_h);
    hash_combine(seed, key.pad_w);
    hash_combine(seed, key.stride_h);
    hash_combine(seed, key.stride_w);
    hash_combine(seed, key.dilation_h);
    hash_combine(seed, key.dilation_w);
    hash_combine(seed, key.groups);
    return seed;
  }
};

std::unordered_map<Dtype, cutlass::library::NumericTypeID> typeconvert_dict = {
    {Dtype::kB1, cutlass::library::NumericTypeID::kB1},
    {Dtype::kU4, cutlass::library::NumericTypeID::kU4},
    {Dtype::kS4, cutlass::library::NumericTypeID::kS4},
    {Dtype::kU8, cutlass::library::NumericTypeID::kU8},
    {Dtype::kS8, cutlass::library::NumericTypeID::kS8},
    {Dtype::kF16, cutlass::library::NumericTypeID::kF16},
    {Dtype::kU16, cutlass::library::NumericTypeID::kU16},
    {Dtype::kF32, cutlass::library::NumericTypeID::kF32}};

cutlass::library::NumericTypeID DtypeConvert(Dtype type) {
  if (typeconvert_dict.find(type) == typeconvert_dict.end()) {
    BLADNN_LOG(FATAL) << "Invalid Dtype";
  }
  return typeconvert_dict[type];
}

cutlass::library::NumericTypeID GetElementCompute(Dtype type) {
  if (type == Dtype::kB1 || type == Dtype::kU4 || type == Dtype::kS4 ||
      type == Dtype::kU8 || type == Dtype::kS8) {
    return cutlass::library::NumericTypeID::kS32;
  } else {
    return cutlass::library::NumericTypeID::kF32;
  }
}

static bool enable_jit_ =
    1;
static bool verbose_ =
    1;

bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, void* c_ptr,
          int c_dim0, int c_dim1, int batch_count, bool a_is_const,
          bool b_is_const, const void* alpha, const void* beta) {
  if (!enable_jit_) {
    return false;
  }
  if (a_dtype != b_dtype ||
      (a_dtype != Dtype::kF16 && a_dtype != Dtype::kF32 &&
       a_dtype != Dtype::kS8) ||
      (c_dtype != Dtype::kF16 && c_dtype != Dtype::kF32 &&
       c_dtype != Dtype::kS8)) {
    return false;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(ctx->stream);
  auto major_a = a_transpose ? cutlass::library::LayoutTypeID::kRowMajor
                             : cutlass::library::LayoutTypeID::kColumnMajor;
  auto major_b = b_transpose ? cutlass::library::LayoutTypeID::kRowMajor
                             : cutlass::library::LayoutTypeID::kColumnMajor;
  auto in_type = DtypeConvert(a_dtype);
  auto out_type = DtypeConvert(c_dtype);
  auto element_compute = GetElementCompute(a_dtype);
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

  const float alpha_val = 1.0f;
  if (alpha == nullptr) {
    alpha = &alpha_val;
  }

  const float beta_val = 0.0f;
  if (beta == nullptr) {
    beta = &beta_val;
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
    cutlass::Status status = handle.gemm_tune(
        mode, n, m, k,
        element_compute,                        // data type of internal
                                                // accumulation
        cutlass::library::NumericTypeID::kF32,  // data type of alpha/beta
                                                // scalars
        alpha,                                  // pointer to alpha scalar
        in_type,                                // data type of A matrix
        major_b,                                // layout of A matrix
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
        batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1, c_dim0 * c_dim1,
        c_dim0 * c_dim1, &tune_config, true);
    if (status != cutlass::Status::kSuccess) {
      tune_config.cta_m = -1;
    }
    if (verbose_) {
      std::cout << "tune_config " << tune_config.cta_m << " "
                << tune_config.cta_n << " " << tune_config.cta_k << " "
                << tune_config.stage << " " << tune_config.split_k << " "
                << std::endl;
    }
    iter = configs.insert(std::make_pair(key, tune_config)).first;
  }

  if (iter == configs.end() || iter->second.cta_m <= 0) {
    // fallback to cublas
    // handle.gemm_cublas(n, m, k, &alpha, in_type, major_b, b_ptr, ldb,
    // in_type,
    //                    major_a, a_ptr, lda, &beta, out_type, c_ptr, ldc);
    return false;
  } else {
    auto tune_config = iter->second;
    cutlass::Status status = handle.gemm_tune(
        mode, n, m, k,
        element_compute,                        // data type of internal
                                                // accumulation
        cutlass::library::NumericTypeID::kF32,  // data type of alpha/beta
                                                // scalars
        alpha,                                  // pointer to alpha scalar
        in_type,                                // data type of A matrix
        major_b,                                // layout of A matrix
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
        batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1, c_dim0 * c_dim1,
        c_dim0 * c_dim1, &tune_config, false);
    return true;
  }

  return false;
}

bool spgemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
            int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
            const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype,
            void* c_ptr, int c_dim0, int c_dim1, Dtype e_dtype,
            const void* e_ptr, int e_dim0, int e_dim1, int batch_count,
            bool a_is_const, bool b_is_const, const void* alpha,
            const void* beta) {
  if (!enable_jit_) {
    return false;
  }
  if (a_dtype != b_dtype ||
      (a_dtype != Dtype::kF16 && a_dtype != Dtype::kF32 &&
       a_dtype != Dtype::kS8) ||
      (c_dtype != Dtype::kF16 && c_dtype != Dtype::kF32 &&
       c_dtype != Dtype::kS8) ||
      (e_dtype != Dtype::kU16)) {
    return false;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(ctx->stream);
  auto major_a = a_transpose ? cutlass::library::LayoutTypeID::kColumnMajor
                             : cutlass::library::LayoutTypeID::kRowMajor;
  auto major_b = b_transpose ? cutlass::library::LayoutTypeID::kColumnMajor
                             : cutlass::library::LayoutTypeID::kRowMajor;
  auto in_type = DtypeConvert(a_dtype);
  auto out_type = DtypeConvert(c_dtype);
  auto indices_type = DtypeConvert(e_dtype);
  auto element_compute = GetElementCompute(a_dtype);
  static std::map<CutlassSparseGemmTuningCacheKey, cutlass::library::TuneConfig>
      configs;
  static cutlass::library::BladeHandle handle;
  static std::mutex mu_;

  int m = a_transpose ? a_dim1 : a_dim0;
  int k = b_transpose ? b_dim1 : b_dim0;
  int n = b_transpose ? b_dim0 : b_dim1;
  int lda = a_dim1;
  int ldb = b_dim1;
  int ldc = c_dim1;
  int lde = e_dim1;

  const float alpha_val = 1.0f;
  if (alpha == nullptr) {
    alpha = &alpha_val;
  }

  const float beta_val = 0.0f;
  if (beta == nullptr) {
    beta = &beta_val;
  }

  CutlassSparseGemmTuningCacheKey key(a_dtype, c_dtype, e_dtype, m, n, k,
                                      a_transpose, b_transpose, batch_count);
  handle.set_stream(stream);
  auto mode = batch_count > 1 ? cutlass::library::GemmUniversalMode::kBatched
                              : cutlass::library::GemmUniversalMode::kGemm;
  auto iter = configs.find(key);
  if (iter == configs.end()) {
    std::lock_guard<std::mutex> l(mu_);
    cutlass::library::TuneConfig tune_config;
    cutlass::Status status = handle.spgemm_tune(
        mode, m, n, k,
        element_compute,                        // data type of internal
                                                // accumulation
        cutlass::library::NumericTypeID::kF32,  // data type of alpha/beta
                                                // scalars
        alpha,                                  // pointer to alpha scalar
        in_type,                                // data type of A matrix
        major_a,                                // layout of A matrix
        cutlass::library::ComplexTransform::kNone,
        a_ptr,    // pointer to A matrix in device memory
        lda,      // leading dimension of A matrix
        in_type,  // data type of B matrix
        major_b,  // layout of B matrix
        cutlass::library::ComplexTransform::kNone,
        b_ptr,         // pointer to B matrix in device memory
        ldb,           // leading dimension of B matrix
        beta,          // pointer to beta scalar
        out_type,      // data type of C and D matrix
        c_ptr,         // pointer to C matrix in device memory
        ldc,           // leading dimension fo C matrix
        c_ptr,         // pointer to D matrix in device memory
        ldc,           // leading dimension of D matrix
        indices_type,  // data type of E matrix
        e_ptr,         // pointer to E matrix in device memory
        lde,           // leading dimension of E matrix
        batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1, c_dim0 * c_dim1,
        c_dim0 * c_dim1, e_dim0 * e_dim1, &tune_config, true);
    if (status != cutlass::Status::kSuccess) {
      tune_config.cta_m = -1;
    }
    if (verbose_) {
      std::cout << "tune_config " << tune_config.cta_m << " "
                << tune_config.cta_n << " " << tune_config.cta_k << " "
                << tune_config.stage << " " << tune_config.split_k << " "
                << std::endl;
    }
    iter = configs.insert(std::make_pair(key, tune_config)).first;
  }

  if (iter == configs.end() || iter->second.cta_m <= 0) {
    // fallback to dense gemm
    return false;
  } else {
    auto tune_config = iter->second;
    cutlass::Status status = handle.spgemm_tune(
        mode, m, n, k,
        element_compute,                        // data type of internal
                                                // accumulation
        cutlass::library::NumericTypeID::kF32,  // data type of alpha/beta
                                                // scalars
        alpha,                                  // pointer to alpha scalar
        in_type,                                // data type of A matrix
        major_a,                                // layout of A matrix
        cutlass::library::ComplexTransform::kNone,
        a_ptr,    // pointer to A matrix in device memory
        lda,      // leading dimension of A matrix
        in_type,  // data type of B matrix
        major_b,  // layout of B matrix
        cutlass::library::ComplexTransform::kNone,
        b_ptr,         // pointer to B matrix in device memory
        ldb,           // leading dimension of B matrix
        beta,          // pointer to beta scalar
        out_type,      // data type of C and D matrix
        c_ptr,         // pointer to C matrix in device memory
        ldc,           // leading dimension fo C matrix
        c_ptr,         // pointer to D matrix in device memory
        ldc,           // leading dimension of D matrix
        indices_type,  // data type of E matrix
        e_ptr,         // pointer to E matrix in device memory
        lde,           // leading dimension of E matrix
        batch_count, b_dim0 * b_dim1, a_dim0 * a_dim1, c_dim0 * c_dim1,
        c_dim0 * c_dim1, e_dim0 * e_dim1, &tune_config, false);
    return true;
  }

  return false;
}

bool conv2d(void* s, Dtype in_dtype, Dtype out_dtype, ConvKind conv_kind,
            Layout data_layout, Layout kernel_layout, int N, int H, int W,
            int C, int K, int R, int S, int P, int Q, int pad_h, int pad_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int groups,

            void const* alpha,  /// Pointer to alpha scalar

            void const* ptr_A,  /// Pointer to A matrix in Global Memory

            void const* ptr_B,  /// Pointer to B matrix in Global Memory

            void const* beta,  /// Pointer to beta scalar

            void const* ptr_C,  /// Pointer to C matrix

            void* ptr_D,  /// Pointer to D matrix

            bool bias_c, Activation activation) {
  cudaStream_t stream = static_cast<cudaStream_t>(s);
  static std::unordered_map<CutlassConv2DTuningCacheKey,
                            cutlass::library::TuneConfig,
                            CutlassConv2DTuningCacheKeyHasher>
      configs;
  static cutlass::library::BladeHandle handle;
  auto in_type = DtypeConvert(in_dtype);
  auto out_type = DtypeConvert(out_dtype);
  auto element_compute = GetElementCompute(in_dtype);
  auto element_scalar = cutlass::library::NumericTypeID::kF32;
  auto kind = cutlass::library::ConvKind::kFprop;
  auto layout_A = cutlass::library::LayoutTypeID::kTensorNHWC;
  auto layout_B = cutlass::library::LayoutTypeID::kTensorNHWC;
  auto layout_C = cutlass::library::LayoutTypeID::kTensorNHWC;
  static std::mutex mu_;
  CutlassConv2DTuningCacheKey key{
      in_dtype, out_dtype, conv_kind, data_layout, kernel_layout,
      N,        H,         W,         C,           K,
      R,        S,         P,         Q,           pad_h,
      pad_w,    stride_h,  stride_w,  dilation_h,  dilation_w,
      groups};
  handle.set_stream(stream);
  bool relu = activation == Activation::kRelu;
  auto iter = configs.find(key);
  if (iter == configs.end() && enable_jit_ == true) {
    std::lock_guard<std::mutex> l(mu_);
    cutlass::library::TuneConfig tune_config;
    cutlass::Status status = handle.conv2d_tune(
        kind, N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, groups,
        element_compute,  /// Data type of internal accumulation

        element_scalar,  /// Data type of alpha/beta scalars

        alpha,  /// Pointer to alpha scalar

        in_type,   /// Data type of A matrix elements
        layout_A,  /// Layout of A matrix
        ptr_A,     /// Pointer to A matrix in Global Memory

        in_type,   /// Data type of B matrix elements
        layout_B,  /// Layout of B matrix
        ptr_B,     /// Pointer to B matrix in Global Memory

        beta,  /// Pointer to beta scalar

        out_type,  /// Data type of C and D matrices
        ptr_C,     /// Pointer to C matrix
        layout_C,  /// Layout of C matrix

        ptr_D,  /// Pointer to D matrix
        &tune_config, true, bias_c, relu);
    if (status != cutlass::Status::kSuccess) {
      tune_config.cta_m = -1;
    }
    if (verbose_) {
      std::cout << "tune_config " << tune_config.cta_m << " "
                << tune_config.cta_n << " " << tune_config.cta_k << " "
                << tune_config.stage << " " << tune_config.split_k << " "
                << std::endl;
    }
    iter = configs.insert(std::make_pair(key, tune_config)).first;
  }
  if (iter == configs.end() || iter->second.cta_m <= 0) {
    // fallback
    return false;
  } else {
    auto tune_config = iter->second;
    cutlass::Status status = handle.conv2d_tune(
        kind, N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, groups,
        element_compute,  /// Data type of internal accumulation

        element_scalar,  /// Data type of alpha/beta scalars

        alpha,  /// Pointer to alpha scalar

        in_type,   /// Data type of A matrix elements
        layout_A,  /// Layout of A matrix
        ptr_A,     /// Pointer to A matrix in Global Memory

        in_type,   /// Data type of B matrix elements
        layout_B,  /// Layout of B matrix
        ptr_B,     /// Pointer to B matrix in Global Memory

        beta,  /// Pointer to beta scalar

        out_type,  /// Data type of C and D matrices
        ptr_C,     /// Pointer to C matrix
        layout_C,  /// Layout of C matrix

        ptr_D,  /// Pointer to D matrix
        &tune_config, false, bias_c, relu);
    if (status == cutlass::Status::kSuccess) {
      return true;
    } else {
      return false;
    }
  }
  return false;
}
}  // namespace bladnn
