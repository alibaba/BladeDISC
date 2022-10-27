#pragma once
#include <cublas_v2.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "bladnn/backend/cutlass/cutlass_sparse.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/util.h"

namespace cutlass {
namespace library {
bool ReadBoolFromEnvVar(const char* name, bool default_value = false);

struct TuneConfig {
  int cta_m;
  int cta_n;
  int cta_k;
  int stage;
  int split_k;
  std::string name;
  TuneConfig(int cta_m = -1, int cta_n = -1, int cta_k = -1, int stage = -1,
             int split_k = -1, std::string name = "")
      : cta_m(cta_m),
        cta_n(cta_n),
        cta_k(cta_k),
        stage(stage),
        split_k(split_k),
        name(name) {}
};

class BladeHandle : public Handle {
 public:
  BladeHandle(cudaStream_t stream = nullptr, size_t workspace_size = (4 << 20))
      : Handle(stream, workspace_size) {
    cublasCreate(&cublas_handle);
    sparse_operation_table_ = new SparseOperationTable();
  }
  ~BladeHandle() {}
  cublasHandle_t cublas_handle;
  Operation const* last_operation_;
  static int const kHostWorkspaceSize = (4 << 10);
  SparseOperationTable* sparse_operation_table_;

  template <typename ConfigureType, typename ArgumentType>
  void tune_gemm_operations(
      std::vector<Operation const*> operations,  /// Operations List
      ConfigureType configuration,               /// Operations Configuration
      ArgumentType arguments,                    /// Operations Arguments
      int M,                                     /// GEMM M dimension
      int N,                                     /// GEMM N dimension
      int K,                                     /// GEMM K dimension
      std::vector<int> split_ks,                 /// Split K search space
      int times,                         /// Iteration number of each operation
      Operation const** best_operation,  /// Final Best operations
      int* best_split_k,                 /// Final Best split k
      float* best_time,                  /// Final Best time
      bool gemm_mode                     /// Whether is GEMM
  );

  Status gemm_cublas(
      int M,                    /// GEMM M dimension
      int N,                    /// GEMM N dimension
      int K,                    /// GEMM K dimension
      void const* alpha,        /// Pointer to alpha scalar
      NumericTypeID element_A,  /// Data type of A matrix elements
      LayoutTypeID layout_A,    /// Layout of A matrix
      void const* ptr_A,        /// Pointer to A matrix in Global Memory
      int64_t lda,              /// Leading dimension of A matrix
      NumericTypeID element_B,  /// Data type of A matrix elements
      LayoutTypeID layout_B,    /// Layout of A matrix
      void const* ptr_B,        /// Pointer to A matrix in Global Memory
      int64_t ldb,              /// Leading dimension of A matrix
      void const* beta,         /// Pointer to alpha scalar
      NumericTypeID element_C,  /// Data type of A matrix elements
      void* ptr_D,              /// Pointer to C matrix
      int64_t ldd);             /// Leading dimension of C matrix

  Status gemm_tune(

      GemmUniversalMode
          mode,  /// indicates the mode in which the kUniversal GEMM is launched

      int M,  /// GEMM M dimension
      int N,  /// GEMM N dimension
      int K,  /// GEMM K dimension

      NumericTypeID element_compute,  /// Data type of internal accumulation

      NumericTypeID element_scalar,  /// Data type of alpha/beta scalars

      void const* alpha,  /// Pointer to alpha scalar

      NumericTypeID element_A,  /// Data type of A matrix elements
      LayoutTypeID layout_A,    /// Layout of A matrix
      ComplexTransform
          transform_A,  /// Complex transformation applied to A matrix - ignored
                        /// for real-valued matrices

      void const* ptr_A,  /// Pointer to A matrix in Global Memory
      int64_t lda,        /// Leading dimension of A matrix

      NumericTypeID element_B,  /// Data type of B matrix elements
      LayoutTypeID layout_B,    /// Layout of B matrix
      ComplexTransform
          transform_B,  /// Complex transformation applied to B matrix - ignored
                        /// for real-valued matrices

      void const* ptr_B,  /// Pointer to B matrix in Global Memory
      int64_t ldb,        /// Leading dimension of B matrix

      void const* beta,  /// Pointer to beta scalar

      NumericTypeID element_C,  /// Data type of C and D matrices

      void const* ptr_C,  /// Pointer to C matrix
      int64_t ldc,        /// Leading dimension of C matrix

      void* ptr_D,  /// Pointer to D matrix
      int64_t ldd,  /// Leading dimension of D matrix

      int batch_count = 1,  /// Batch count or number of split-K slices

      int64_t batch_stride_A = 0,  /// Batch stride of A operand
      int64_t batch_stride_B = 0,  /// Batch stride of B operand
      int64_t batch_stride_C = 0,  /// Batch stride of C operand
      int64_t batch_stride_D = 0,  /// Batch stride of D operand

      TuneConfig* tune_config = nullptr, bool tune = false);

  Status spgemm_tune(

      GemmUniversalMode
          mode,  /// indicates the mode in which the kUniversal GEMM is launched

      int M,  /// GEMM M dimension
      int N,  /// GEMM N dimension
      int K,  /// GEMM K dimension

      NumericTypeID element_compute,  /// Data type of internal accumulation

      NumericTypeID element_scalar,  /// Data type of alpha/beta scalars

      void const* alpha,  /// Pointer to alpha scalar

      NumericTypeID element_A,  /// Data type of A matrix elements
      LayoutTypeID layout_A,    /// Layout of A matrix
      ComplexTransform
          transform_A,  /// Complex transformation applied to A matrix - ignored
                        /// for real-valued matrices

      void const* ptr_A,  /// Pointer to A matrix in Global Memory
      int64_t lda,        /// Leading dimension of A matrix

      NumericTypeID element_B,  /// Data type of B matrix elements
      LayoutTypeID layout_B,    /// Layout of B matrix
      ComplexTransform
          transform_B,  /// Complex transformation applied to B matrix - ignored
                        /// for real-valued matrices

      void const* ptr_B,  /// Pointer to B matrix in Global Memory
      int64_t ldb,        /// Leading dimension of B matrix

      void const* beta,  /// Pointer to beta scalar

      NumericTypeID element_C,  /// Data type of C and D matrices

      void const* ptr_C,  /// Pointer to C matrix
      int64_t ldc,        /// Leading dimension of C matrix

      void* ptr_D,  /// Pointer to D matrix
      int64_t ldd,  /// Leading dimension of D matrix

      NumericTypeID element_E,  /// Data type of E matrices

      void const* ptr_E,  /// Pointer to E matrix
      int64_t lde,        /// Leading dimension of E matrix

      int batch_count = 1,  /// Batch count or number of split-K slices

      int64_t batch_stride_A = 0,  /// Batch stride of A operand
      int64_t batch_stride_B = 0,  /// Batch stride of B operand
      int64_t batch_stride_C = 0,  /// Batch stride of C operand
      int64_t batch_stride_D = 0,  /// Batch stride of D operand
      int64_t batch_stride_E = 0,  /// Batch stride of E operand

      TuneConfig* tune_config = nullptr, bool tune = false);

  Status conv2d_tune(
      ConvKind kind, int N, int H, int W, int C, int K, int R, int S, int P,
      int Q, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
      int dilation_w, int groups,
      NumericTypeID element_compute,  /// Data type of internal accumulation

      NumericTypeID element_scalar,  /// Data type of alpha/beta scalars

      void const* alpha,  /// Pointer to alpha scalar

      NumericTypeID element_A,  /// Data type of A matrix elements
      LayoutTypeID layout_A,    /// Layout of A matrix
      void const* ptr_A,        /// Pointer to A matrix in Global Memory

      NumericTypeID element_B,  /// Data type of B matrix elements
      LayoutTypeID layout_B,    /// Layout of B matrix
      void const* ptr_B,        /// Pointer to B matrix in Global Memory

      void const* beta,  /// Pointer to beta scalar

      NumericTypeID element_C,  /// Data type of C and D matrices
      void const* ptr_C,        /// Pointer to C matrix
      LayoutTypeID layout_C,    /// Layout of C matrix

      void* ptr_D,  /// Pointer to D matrix
      TuneConfig* tune_config, bool tune, bool bias_c = false,
      bool relu = false);
};
}  // namespace library
}  // namespace cutlass
