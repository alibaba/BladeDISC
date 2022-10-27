#include "bladnn/backend/cutlass/cutlass_handle.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "bladnn/utils/log.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/util.h"

#define checkCudaErrors(val) \
  if (val != cudaSuccess) exit(EXIT_FAILURE)
#define checkCUDNN(status)                                        \
  do {                                                            \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      BLADNN_LOG(FATAL) << (_error.str());                        \
    }                                                             \
  } while (0)

namespace cutlass {
namespace library {
bool ReadBoolFromEnvVar(const char* name, bool default_value) {
  const char* env = std::getenv(name);
  if (!env) return default_value;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

/// Returns the maximum required alignment for each operator
template <typename Description>
static int maximum_alignment_requirement(Description const& desc) {
  return std::max(std::max(desc.A.alignment, desc.B.alignment),
                  desc.C.alignment);
}

/// Returns the largest alignment (in units of elements) the problem satisfies,
/// starting from a given upper limit.
static int gemm_problem_alignment(
    int M, int N, int K, NumericTypeID element_A, void const* ptr_A,
    int64_t lda, int64_t batch_stride_A, NumericTypeID element_B,
    void const* ptr_B, int64_t ldb, int64_t batch_stride_B,
    NumericTypeID element_C, void const* ptr_C, int64_t ldc,
    int64_t batch_stride_C, void const* ptr_D, int64_t ldd,
    int64_t batch_stride_D, int max_alignment_in_bytes = 16) {
  void const* pointers[] = {ptr_A, ptr_B, ptr_C, ptr_D};

  int64_t extents[] = {M,
                       N,
                       K,
                       lda,
                       ldb,
                       ldc,
                       ldd,
                       batch_stride_A,
                       batch_stride_B,
                       batch_stride_C,
                       batch_stride_D};

  NumericTypeID elements[] = {element_A, element_B, element_C};

  for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
    bool satisfied = true;

    // Can pointers satisfy this?
    for (void const* ptr : pointers) {
      std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);

      if (int_ptr % max_alignment_in_bytes) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Compute the maximum alignment based on element data types
    int max_element_alignment = 0;

    for (NumericTypeID type_id : elements) {
      int element_alignment =
          max_alignment_in_bytes * 8 / library::sizeof_bits(type_id);
      max_element_alignment =
          std::max(max_element_alignment, element_alignment);
    }

    // Can the problem size and leading dimensions satisfy this?
    for (int64_t extent : extents) {
      if (extent % max_element_alignment) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Yes
    return max_element_alignment;
  }

  // No alignment satisfies this problem
  return 0;
}

static int conv_problem_alignment(NumericTypeID element_A, void const* ptr_A,
                                  int64_t lda, NumericTypeID element_B,
                                  void const* ptr_B, int64_t ldb,
                                  NumericTypeID element_C, void const* ptr_C,
                                  int64_t ldc, void const* ptr_D, int64_t ldd,
                                  int max_alignment_in_bytes = 16) {
  void const* pointers[] = {ptr_A, ptr_B, ptr_C, ptr_D};

  int64_t extents[] = {lda, ldb, ldc, ldd};

  NumericTypeID elements[] = {element_A, element_B, element_C};

  for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
    bool satisfied = true;

    // Can pointers satisfy this?
    for (void const* ptr : pointers) {
      std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);

      if (int_ptr % max_alignment_in_bytes) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Compute the maximum alignment based on element data types
    int max_element_alignment = 0;

    for (NumericTypeID type_id : elements) {
      int element_alignment =
          max_alignment_in_bytes * 8 / library::sizeof_bits(type_id);
      max_element_alignment =
          std::max(max_element_alignment, element_alignment);
    }

    // Can the problem size and leading dimensions satisfy this?
    for (int64_t extent : extents) {
      if (extent % max_element_alignment) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Yes
    return max_element_alignment;
  }

  // No alignment satisfies this problem
  return 0;
}

template <typename FunctionalMapIter, typename PerferenceKey,
          typename Description>
static void find_gemm_operations(FunctionalMapIter operators_it,
                                 PerferenceKey const preference_key,
                                 std::vector<Operation const*>* operations,
                                 TuneConfig* tune_config = nullptr) {
  auto cc_it = operators_it->second.upper_bound(preference_key);

  if (cc_it == operators_it->second.begin()) {
    return;
  }

  // Search in descending order of compute capability
  do {
    --cc_it;

    // Search tile sizes in order, for now.
    if (cc_it->first.alignment > preference_key.alignment) {
      continue;
    }
    for (auto const* op : cc_it->second) {
      Description const& desc =
          static_cast<Description const&>(op->description());
      // std::cout << "Description " << desc.name << std::endl;

      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;

      int op_alignment = maximum_alignment_requirement<Description>(desc);

      if ((min_cc <= preference_key.compute_capability) &&
          (preference_key.compute_capability <= max_cc) &&
          (op_alignment <= preference_key.alignment)) {
        auto threadblock_shape = desc.tile_description.threadblock_shape;
        if (tune_config) {
          // std::cout << "tune_config->name " << tune_config->name <<
          // std::endl;
          if (threadblock_shape.m() == tune_config->cta_m &&
              threadblock_shape.n() == tune_config->cta_n &&
              threadblock_shape.k() == tune_config->cta_k &&
              desc.tile_description.threadblock_stages == tune_config->stage) {
            operations->push_back(op);
            break;
          }
        } else {
          operations->push_back(op);
        }
      }
    }
  } while (operations->empty() && cc_it != operators_it->second.begin());

  return;
}

static void find_conv_operations(
    ConvOperationFunctionalMap::const_iterator operators_it,
    ConvPreferenceKey const preference_key, const int alignment,
    std::vector<Operation const*>* operations,
    TuneConfig* tune_config = nullptr) {
  auto cc_it = operators_it->second.upper_bound(preference_key);

  if (cc_it == operators_it->second.begin()) {
    return;
  }

  // Search in descending order of compute capability
  do {
    --cc_it;

    // Search tile sizes in order, for now.
    for (auto const* op : cc_it->second) {
      ConvDescription const& desc =
          static_cast<ConvDescription const&>(op->description());

      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;
      int op_alignment = maximum_alignment_requirement<ConvDescription>(desc);

      if ((min_cc <= preference_key.compute_capability) &&
          (preference_key.compute_capability <= max_cc) &&
          (op_alignment <= alignment)) {
        auto threadblock_shape = desc.tile_description.threadblock_shape;
        if (tune_config) {
          if (threadblock_shape.m() == tune_config->cta_m &&
              threadblock_shape.n() == tune_config->cta_n &&
              threadblock_shape.k() == tune_config->cta_k &&
              desc.tile_description.threadblock_stages == tune_config->stage) {
            // std::cout << "ConvDescription " << desc.name << std::endl;
            operations->push_back(op);
          }
        } else {
          operations->push_back(op);
        }
      }
    }
  } while (operations->empty() && cc_it != operators_it->second.begin());

  return;
}

std::unordered_map<cutlass::library::NumericTypeID, cudnnDataType_t>
    typeconvert_dict = {
        {cutlass::library::NumericTypeID::kS8, CUDNN_DATA_INT8},
        {cutlass::library::NumericTypeID::kF16, CUDNN_DATA_HALF},
        {cutlass::library::NumericTypeID::kF32, CUDNN_DATA_FLOAT}};

cudnnDataType_t DtypeConvert(cutlass::library::NumericTypeID type) {
  if (typeconvert_dict.find(type) == typeconvert_dict.end()) {
    BLADNN_LOG(FATAL) << "Invalid Dtype";
  }
  return typeconvert_dict[type];
}

static bool verbose_ = 1;

template <typename ConfigureType, typename ArgumentType>
void BladeHandle::tune_gemm_operations(
    std::vector<Operation const*> operations, ConfigureType configuration,
    ArgumentType arguments, int M, int N, int K, std::vector<int> split_ks,
    int times, Operation const** best_operation, int* best_split_k,
    float* best_time, bool gemm_mode) {
  *best_operation = operations[0];
  *best_split_k = 2;
  *best_time = -1.0;
  for (auto split_k : split_ks) {
    if (gemm_mode) {
      configuration.batch_count = split_k;
    }
    for (auto operation : operations) {
      // Query host work space size
      uint64_t host_workspace_size_needed =
          operation->get_host_workspace_size(&configuration);

      if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
        if (verbose_) {
          std::cout << "host_workspace_size_needed="
                    << host_workspace_size_needed << " > " << kHostWorkspaceSize
                    << std::endl;
        }
        continue;
      }

      char host_workspace[kHostWorkspaceSize];

      // Query device workspace size
      uint64_t device_workspace_size_needed =
          operation->get_device_workspace_size(&configuration, &arguments);

      if (uint64_t(get_workspace_size()) < device_workspace_size_needed) {
        if (verbose_) {
          std::cout << "device_workspace_size_needed="
                    << device_workspace_size_needed << " > "
                    << get_workspace_size() << std::endl;
        }
        continue;
      }

      // Initialize host and device workspaces
      Status status = operation->initialize(&configuration, host_workspace,
                                            get_workspace(), get_stream());

      if (status != cutlass::Status::kSuccess) {
        if (verbose_) {
          std::cout << "initialize failed" << std::endl;
        }
        continue;
      }

      // Run the operator
      status = operation->run(&arguments, host_workspace, get_workspace(),
                              get_stream());

      if (status != cutlass::Status::kSuccess) {
        if (verbose_) {
          std::cout << "run failed" << std::endl;
        }
        continue;
      }
      cudaEvent_t start, stop;
      checkCudaErrors(cudaEventCreate(&start));
      checkCudaErrors(cudaEventCreate(&stop));
      checkCudaErrors(cudaEventRecord(start));
      for (int i = 0; i < times; ++i) {
        operation->run(&arguments, host_workspace, get_workspace(),
                       get_stream());
      }
      checkCudaErrors(cudaEventRecord(stop));
      checkCudaErrors(cudaEventSynchronize(stop));
      float msec = 0.0f;
      checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
      float time = 1000 * msec / times;
      GemmDescription const& desc =
          static_cast<GemmDescription const&>(operation->description());
      auto threadblock_shape = desc.tile_description.threadblock_shape;
      if (verbose_) {
        std::cout << "time: " << time << " us"
                  << " Config:"
                  << " " << threadblock_shape.m() << " "
                  << threadblock_shape.n() << " " << threadblock_shape.k()
                  << " " << desc.tile_description.threadblock_stages << " "
                  << split_k << " " << desc.name << std::endl;
      }
      if (time > 0 && (*best_time < 0 || time < *best_time)) {
        *best_time = time;
        *best_operation = operation;
        *best_split_k = split_k;
      }
    }  // end for operation
  }    // end for split_k
}

Status BladeHandle::gemm_cublas(
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
    int64_t ldd               /// Leading dimension of C matrix
) {
  cublasGemmAlgo_t tensor_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  cublasSetStream(cublas_handle, get_stream());
  auto type_a = (element_A == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
  auto type_b = (element_B == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
  auto type_c = (element_C == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
  int device_major = compute_capability() / 10;
  auto compute_type = (element_A == NumericTypeID::kF16 || device_major < 8)
                          ? CUBLAS_COMPUTE_32F
                          : CUBLAS_COMPUTE_32F_FAST_TF32;
  auto tp_a =
      (layout_A == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto tp_b =
      (layout_B == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasGemmEx(cublas_handle, tp_a, tp_b, M, N, K, alpha, ptr_A, type_a, lda,
               ptr_B, type_b, ldb, beta, ptr_D, type_c, ldd, compute_type,
               tensor_algo);
  return cutlass::Status::kSuccess;
}

float profile_cublasLt(cublasOperation_t transa, cublasOperation_t transb,
                       int m, int n, int k, void const* A, int lda,
                       void const* B, int ldb, void* C, int ldc,
                       void const* alpha, void const* beta, int nIter) {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb, sizeof(transb));

  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, transa == CUBLAS_OP_N ? m : k,
                             transa == CUBLAS_OP_N ? k : m, lda);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, transb == CUBLAS_OP_N ? k : n,
                             transb == CUBLAS_OP_N ? n : k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, ldc);

  // no need to transform C matrix as beta is assumed to be 0
  cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C,
                 Cdesc, C, Cdesc, NULL, NULL, 0, 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float msecTotal = 0;

  cudaEventRecord(start);
  for (int i = 0; i < nIter; i += 1) {
    cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C,
                   Cdesc, C, Cdesc, NULL, NULL, 0, 0);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);

  float usecPerMatrixMul = 1000 * msecTotal / nIter;

  if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);

  // wait until device is done before freeing transformed buffers
  cudaDeviceSynchronize();
  return usecPerMatrixMul;
}

Status BladeHandle::gemm_tune(

    GemmUniversalMode
        mode,  /// indicates the mode in which the kUniversal GEMM is launched

    int M,  /// GEMM M dimension
    int N,  /// GEMM N dimension
    int K,  /// GEMM K dimension

    NumericTypeID element_compute,  /// Data type of internal accumulation

    NumericTypeID element_scalar,  /// Data type of alpha/beta scalars

    void const* alpha,  /// Pointer to alpha scalar

    NumericTypeID element_A,       /// Data type of A matrix elements
    LayoutTypeID layout_A,         /// Layout of A matrix
    ComplexTransform transform_A,  /// Complex transformation applied to A
                                   /// matrix - ignored for real-valued matrices

    void const* ptr_A,  /// Pointer to A matrix in Global Memory
    int64_t lda,        /// Leading dimension of A matrix

    NumericTypeID element_B,       /// Data type of B matrix elements
    LayoutTypeID layout_B,         /// Layout of B matrix
    ComplexTransform transform_B,  /// Complex transformation applied to B
                                   /// matrix - ignored for real-valued matrices

    void const* ptr_B,  /// Pointer to B matrix in Global Memory
    int64_t ldb,        /// Leading dimension of B matrix

    void const* beta,  /// Pointer to beta scalar

    NumericTypeID element_C,  /// Data type of C and D matrices

    void const* ptr_C,  /// Pointer to C matrix
    int64_t ldc,        /// Leading dimension of C matrix

    void* ptr_D,  /// Pointer to D matrix
    int64_t ldd,  /// Leading dimension of D matrix

    int batch_count,  /// Batch count or number of split-K slices

    int64_t batch_stride_A,  /// Batch stride of A operand
    int64_t batch_stride_B,  /// Batch stride of B operand
    int64_t batch_stride_C,  /// Batch stride of C operand
    int64_t batch_stride_D,  /// Batch stride of D operand

    TuneConfig* tune_config, bool tune) {
  //
  // Find the operation
  //

  auto tuning_begin = std::chrono::steady_clock::now();
  GemmFunctionalKey key(get_provider(), GemmKind::kUniversal, element_compute,
                        element_scalar, element_A, layout_A, transform_A,
                        element_B, layout_B, transform_B, element_C);

  auto operators_it =
      Singleton::get().operation_table.gemm_operations.find(key);

  if (operators_it == Singleton::get().operation_table.gemm_operations.end()) {
    return cutlass::Status::kErrorNotSupported;
  }

  if (operators_it->second.empty()) {
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Compute the largest alignment restriction the kernel can satisfy.
  //

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;

  void const* ptr_A_check = ptr_A;
  void const* ptr_B_check = ptr_B;
  void const* ptr_C_check = ptr_C;
  void* ptr_D_check = ptr_D;

  // Ignore alignment of pointers to pointers. We can't check this from the
  // host, as each batch index has its own pointer in device memory.
  if (mode == GemmUniversalMode::kArray) {
    ptr_A_check = nullptr;
    ptr_B_check = nullptr;
    ptr_C_check = nullptr;
    ptr_D_check = nullptr;
  }

  int alignment =
      gemm_problem_alignment(M, N, K, element_A, ptr_A_check, lda, 0, element_B,
                             ptr_B_check, ldb, 0, element_C, ptr_C_check, ldc,
                             0, ptr_D_check, ldd, 0, kMaximumAlignmentSize);
  // std::cout << "gemm_problem_alignment " << alignment << std::endl;

  //
  // Find the best kernel in descending order of preference.
  //

  GemmPreferenceKey preference_key(compute_capability(), alignment);

  std::vector<Operation const*> operations;
  if (tune) {
    find_gemm_operations<GemmOperationFunctionalMap::const_iterator,
                         GemmPreferenceKey, GemmDescription>(
        operators_it, preference_key, &operations);
  } else {
    find_gemm_operations<GemmOperationFunctionalMap::const_iterator,
                         GemmPreferenceKey, GemmDescription>(
        operators_it, preference_key, &operations, tune_config);
  }

  if (operations.empty()) {
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Configure operation
  //

  GemmUniversalConfiguration configuration{mode, {M, N, K}, batch_count, lda,
                                           ldb,  ldc,       ldd};

  GemmUniversalArguments arguments{ptr_A,
                                   ptr_B,
                                   ptr_C,
                                   ptr_D,
                                   alpha,
                                   beta,
                                   get_scalar_pointer_mode(),
                                   batch_stride_A,
                                   batch_stride_B,
                                   batch_stride_C,
                                   batch_stride_D};

  bool gemm_mode = (mode == GemmUniversalMode::kGemm);
  if (tune) {
    Operation const* best_operation = operations[0];
    int best_split_k = 2;
    float best_time = -1.0;
    std::vector<int> split_ks = {1};
    if (gemm_mode) {
      split_ks = {1, 2, 4, 8, 16};
    }
    int times = 80;
    if ((int64_t)M * N * K >= 2048LL * 2048 * 2048) {
      times = 2;
    } else if ((int64_t)M * N * K >= 1024 * 1024 * 1024) {
      times = 10;
    } else if ((int64_t)M * N * K >= 512 * 512 * 512) {
      times = 40;
    }

    tune_gemm_operations<GemmUniversalConfiguration, GemmUniversalArguments>(
        operations, configuration, arguments, M, N, K, split_ks, times,
        &best_operation, &best_split_k, &best_time, gemm_mode);

    TuneConfig best_config;
    GemmDescription const& desc =
        static_cast<GemmDescription const&>(best_operation->description());
    auto threadblock_shape = desc.tile_description.threadblock_shape;
    best_config.cta_m = threadblock_shape.m();
    best_config.cta_n = threadblock_shape.n();
    best_config.cta_k = threadblock_shape.k();
    best_config.stage = desc.tile_description.threadblock_stages;
    best_config.split_k = best_split_k;
    best_config.name = desc.name;
    *tune_config = best_config;
    if (verbose_) {
      std::cout << "Batch=" << batch_count << " M=" << N << " N=" << M
                << " K=" << K << std::endl;
      std::cout << "best_config " << best_config.cta_m << " "
                << best_config.cta_n << " " << best_config.cta_k << " "
                << best_config.stage << " " << best_config.split_k << " "
                << best_config.name << " " << std::endl;
      std::cout << "best time " << (float)best_time << " us" << std::endl;
    }
    last_operation_ = best_operation;
    if (best_time < 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    int device_major = compute_capability() / 10;
    if (device_major >= 8) {
      float cublas_time;
      if (element_A == cutlass::library::NumericTypeID::kS8 && gemm_mode) {
        // If data type is int8, run cublaslt kernel pass
        auto tp_a = (layout_A == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N
                                                             : CUBLAS_OP_T;
        auto tp_b = (layout_B == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N
                                                             : CUBLAS_OP_T;
        cublas_time = profile_cublasLt(tp_a, tp_b, M, N, K, ptr_A, lda, ptr_B,
                                       ldb, ptr_D, ldd, alpha, beta, times * 2);
      } else {
        cublasGemmAlgo_t tensor_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        cublasSetStream(cublas_handle, get_stream());
        auto type_a =
            (element_A == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
        auto type_b =
            (element_B == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
        auto type_c =
            (element_C == NumericTypeID::kF16) ? CUDA_R_16F : CUDA_R_32F;
        auto compute_type =
            (element_A == NumericTypeID::kF16 || device_major < 8)
                ? CUBLAS_COMPUTE_32F
                : CUBLAS_COMPUTE_32F_FAST_TF32;
        auto tp_a = (layout_A == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N
                                                             : CUBLAS_OP_T;
        auto tp_b = (layout_B == LayoutTypeID::kColumnMajor) ? CUBLAS_OP_N
                                                             : CUBLAS_OP_T;
        if (gemm_mode) {
          cublasGemmEx(cublas_handle, tp_a, tp_b, M, N, K, alpha, ptr_A, type_a,
                       lda, ptr_B, type_b, ldb, beta, ptr_D, type_c, ldd,
                       compute_type, tensor_algo);
        } else {
          cublasGemmStridedBatchedEx(cublas_handle, tp_a, tp_b, M, N, K, alpha,
                                     ptr_A, type_a, lda, batch_stride_A, ptr_B,
                                     type_b, ldb, batch_stride_B, beta, ptr_D,
                                     type_c, ldd, batch_stride_D, batch_count,
                                     compute_type, tensor_algo);
        }

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start));
        for (int i = 0; i < times * 2; ++i) {
          if (gemm_mode) {
            cublasGemmEx(cublas_handle, tp_a, tp_b, M, N, K, alpha, ptr_A,
                         type_a, lda, ptr_B, type_b, ldb, beta, ptr_D, type_c,
                         ldd, compute_type, tensor_algo);
          } else {
            cublasGemmStridedBatchedEx(
                cublas_handle, tp_a, tp_b, M, N, K, alpha, ptr_A, type_a, lda,
                batch_stride_A, ptr_B, type_b, ldb, batch_stride_B, beta, ptr_D,
                type_c, ldd, batch_stride_D, batch_count, compute_type,
                tensor_algo);
          }
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float msec = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
        cublas_time = 1000 * msec / times / 2;
      }

      auto end = std::chrono::steady_clock::now();
      auto tuning_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             end - tuning_begin)
                             .count();
      if (verbose_) {
        std::cout << "cublas time " << (float)cublas_time << " us" << std::endl;
        std::cout << "tuning time " << tuning_time / 1000 << " ms" << std::endl;
      }
      if (cublas_time < best_time) {
        return cutlass::Status::kErrorNotSupported;
      }
    }
    return cutlass::Status::kSuccess;
  }  // end if

  // replay
  if (gemm_mode) {
    configuration.batch_count = tune_config->split_k;
  }
  Operation const* operation = operations[0];
  // Query host work space size
  uint64_t host_workspace_size_needed =
      operation->get_host_workspace_size(&configuration);

  if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
    return cutlass::Status::kErrorNotSupported;
  }

  char host_workspace[kHostWorkspaceSize];

  // Query device workspace size
  uint64_t device_workspace_size_needed =
      operation->get_device_workspace_size(&configuration, &arguments);

  if (uint64_t(get_workspace_size()) < device_workspace_size_needed) {
    return cutlass::Status::kErrorNotSupported;
  }

  // Initialize host and device workspaces
  Status status = operation->initialize(&configuration, host_workspace,
                                        get_workspace(), get_stream());

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  // Run the operator
  return operation->run(&arguments, host_workspace, get_workspace(),
                        get_stream());
}

Status BladeHandle::spgemm_tune(

    GemmUniversalMode
        mode,  /// indicates the mode in which the kUniversal GEMM is launched

    int M,  /// GEMM M dimension
    int N,  /// GEMM N dimension
    int K,  /// GEMM K dimension

    NumericTypeID element_compute,  /// Data type of internal accumulation

    NumericTypeID element_scalar,  /// Data type of alpha/beta scalars

    void const* alpha,  /// Pointer to alpha scalar

    NumericTypeID element_A,       /// Data type of A matrix elements
    LayoutTypeID layout_A,         /// Layout of A matrix
    ComplexTransform transform_A,  /// Complex transformation applied to A
                                   /// matrix - ignored for real-valued matrices

    void const* ptr_A,  /// Pointer to A matrix in Global Memory
    int64_t lda,        /// Leading dimension of A matrix

    NumericTypeID element_B,       /// Data type of B matrix elements
    LayoutTypeID layout_B,         /// Layout of B matrix
    ComplexTransform transform_B,  /// Complex transformation applied to B
                                   /// matrix - ignored for real-valued matrices

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

    int batch_count,  /// Batch count or number of split-K slices

    int64_t batch_stride_A,  /// Batch stride of A operand
    int64_t batch_stride_B,  /// Batch stride of B operand
    int64_t batch_stride_C,  /// Batch stride of C operand
    int64_t batch_stride_D,  /// Batch stride of D operand
    int64_t batch_stride_E,  /// Batch stride of E operand

    TuneConfig* tune_config, bool tune) {
  //
  // Find the operation
  //
  auto tuning_begin = std::chrono::steady_clock::now();
  SparseGemmFunctionalKey key(get_provider(), GemmKind::kSparse,
                              element_compute, element_scalar, element_A,
                              layout_A, transform_A, element_B, layout_B,
                              transform_B, element_C, element_E);
  auto operators_it = sparse_operation_table_->sparse_gemm_operations.find(key);

  if (operators_it == sparse_operation_table_->sparse_gemm_operations.end()) {
    return cutlass::Status::kErrorNotSupported;
  }

  if (operators_it->second.empty()) {
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Compute the largest alignment restriction the kernel can satisfy.
  //

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;

  void const* ptr_A_check = ptr_A;
  void const* ptr_B_check = ptr_B;
  void const* ptr_C_check = ptr_C;
  void* ptr_D_check = ptr_D;
  void const* ptr_E_check = ptr_E;

  // Ignore alignment of pointers to pointers. We can't check this from the
  // host, as each batch index has its own pointer in device memory.
  if (mode == GemmUniversalMode::kArray) {
    ptr_A_check = nullptr;
    ptr_B_check = nullptr;
    ptr_C_check = nullptr;
    ptr_D_check = nullptr;
    ptr_E_check = nullptr;
  }

  int alignment = spgemm_problem_alignment(
      M, N, K, element_A, ptr_A_check, lda, 0, element_B, ptr_B_check, ldb, 0,
      element_C, ptr_C_check, ldc, 0, ptr_D_check, ldd, 0, element_E,
      ptr_E_check, lde, 0, kMaximumAlignmentSize);
  //
  // Find the best kernel in descending order of preference.
  //

  SparseGemmPreferenceKey preference_key(compute_capability(), alignment);

  std::vector<Operation const*> operations;
  if (tune) {
    find_gemm_operations<SparseGemmOperationFunctionalMap::const_iterator,
                         SparseGemmPreferenceKey, SparseGemmDescription>(
        operators_it, preference_key, &operations);
  } else {
    find_gemm_operations<SparseGemmOperationFunctionalMap::const_iterator,
                         SparseGemmPreferenceKey, SparseGemmDescription>(
        operators_it, preference_key, &operations, tune_config);
  }

  if (operations.empty()) {
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Configure operation
  //

  SparseGemmConfiguration configuration{mode,
                                        {M, N, K},
                                        batch_count,
                                        lda,
                                        ldb,
                                        ldc,
                                        ldd,
                                        lde,
                                        batch_stride_A,
                                        batch_stride_B,
                                        batch_stride_C,
                                        batch_stride_D,
                                        batch_stride_E};

  SparseGemmArguments arguments{ptr_A, ptr_B, ptr_C, ptr_D,
                                ptr_E, alpha, beta,  get_scalar_pointer_mode()};

  bool gemm_mode = (mode == GemmUniversalMode::kGemm);
  if (tune) {
    Operation const* best_operation = operations[0];
    int best_split_k = 2;
    float best_time = -1.0;
    std::vector<int> split_ks = {1};
    if (gemm_mode) {
      split_ks = {1, 2, 4, 8, 16};
    }
    int times = 80;
    if ((int64_t)M * N * K >= 2048LL * 2048 * 2048) {
      times = 2;
    } else if ((int64_t)M * N * K >= 1024 * 1024 * 1024) {
      times = 10;
    } else if ((int64_t)M * N * K >= 512 * 512 * 512) {
      times = 40;
    }

    tune_gemm_operations<SparseGemmConfiguration, SparseGemmArguments>(
        operations, configuration, arguments, M, N, K, split_ks, times,
        &best_operation, &best_split_k, &best_time, gemm_mode);

    TuneConfig best_config;
    SparseGemmDescription const& desc =
        static_cast<SparseGemmDescription const&>(
            best_operation->description());
    auto threadblock_shape = desc.tile_description.threadblock_shape;
    best_config.cta_m = threadblock_shape.m();
    best_config.cta_n = threadblock_shape.n();
    best_config.cta_k = threadblock_shape.k();
    best_config.stage = desc.tile_description.threadblock_stages;
    best_config.split_k = best_split_k;
    best_config.name = desc.name;
    *tune_config = best_config;
    if (verbose_) {
      std::cout << "Batch=" << batch_count << " M=" << N << " N=" << M
                << " K=" << K << std::endl;
      std::cout << "best_config " << best_config.cta_m << " "
                << best_config.cta_n << " " << best_config.cta_k << " "
                << best_config.stage << " " << best_config.split_k << " "
                << best_config.name << " " << std::endl;
      std::cout << "best time " << (float)best_time << " us" << std::endl;
    }
    last_operation_ = best_operation;
    if (best_time < 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kSuccess;
  }  // end if

  // replay
  if (gemm_mode) {
    configuration.batch_count = tune_config->split_k;
  }
  Operation const* operation = operations[0];
  // Query host work space size
  uint64_t host_workspace_size_needed =
      operation->get_host_workspace_size(&configuration);

  if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
    return cutlass::Status::kErrorNotSupported;
  }

  char host_workspace[kHostWorkspaceSize];

  // Query device workspace size
  uint64_t device_workspace_size_needed =
      operation->get_device_workspace_size(&configuration, &arguments);

  if (uint64_t(get_workspace_size()) < device_workspace_size_needed) {
    return cutlass::Status::kErrorNotSupported;
  }

  // Initialize host and device workspaces
  Status status = operation->initialize(&configuration, host_workspace,
                                        get_workspace(), get_stream());

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  // Run the operator
  return operation->run(&arguments, host_workspace, get_workspace(),
                        get_stream());
}

Status BladeHandle::conv2d_tune(
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
    TuneConfig* tune_config, bool tune, bool bias_c, bool relu) {
  auto tuning_begin = std::chrono::steady_clock::now();
  int times = 80;
  int GEMM_M = N * P * Q;
  int GEMM_K = R * S * C;
  int GEMM_N = K;
  if ((int64_t)GEMM_M * GEMM_N * GEMM_K >= 2048LL * 2048 * 2048) {
    times = 2;
  } else if ((int64_t)GEMM_M * GEMM_N * GEMM_K >= 1024 * 1024 * 1024) {
    times = 10;
  } else if ((int64_t)GEMM_M * GEMM_N * GEMM_K >= 512 * 512 * 512) {
    times = 40;
  }

  // In conv, element_compute is for epilogue computation, should be kF32 sine
  // alpha and beta are float.
  ConvFunctionalKey key(library::Provider::kCUTLASS, kind, element_A, layout_A,
                        element_B, layout_B, element_C, layout_C,
                        element_compute, cutlass::library::NumericTypeID::kF32);

  auto conv_operations = Singleton::get().operation_table.conv2d_operations;
  if (relu) {
    conv_operations = Singleton::get().operation_table.conv2d_relu_operations;
  }
  auto operators_it = conv_operations.find(key);

  if (operators_it == conv_operations.end()) {
    std::cout << "cutlass_handle: 1056";
    return cutlass::Status::kErrorNotSupported;
  }

  if (operators_it->second.empty()) {
    std::cout << "cutlass_handle: 1061";
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Compute the largest alignment restriction the kernel can satisfy.
  //

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;

  int alignment = conv_problem_alignment(element_A, ptr_A, C, element_B, ptr_B,
                                         C, element_C, ptr_C, K, ptr_D, K,
                                         kMaximumAlignmentSize);

  ConvPreferenceKey preference_key(compute_capability(),
                                   IteratorAlgorithmID::kOptimized);

  std::vector<Operation const*> operations;
  if (tune) {
    find_conv_operations(operators_it, preference_key, alignment, &operations);
  } else {
    find_conv_operations(operators_it, preference_key, alignment, &operations,
                         tune_config);
  }

  if (operations.empty()) {
    std::cout << "cutlass_handle: 1088";
    return cutlass::Status::kErrorNotSupported;
  }

  //
  // Configure operation
  //
  conv::Conv2dProblemSize problem_size(
      N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, conv::Mode::kCrossCorrelation,
      1,  //  split_k
      groups);
  std::vector<int64_t> stride_activations(3);
  std::vector<int64_t> stride_filters(3);
  std::vector<int64_t> stride_output(3);
  stride_activations[0] = (int(problem_size.C));
  stride_activations[1] = (int(problem_size.W) * int(problem_size.C));
  stride_activations[2] =
      (int(problem_size.H) * int(problem_size.W) * int(problem_size.C));

  stride_filters[0] = (int(problem_size.C));
  stride_filters[1] = (int(problem_size.S) * int(problem_size.C));
  stride_filters[2] =
      (int(problem_size.R) * int(problem_size.S) * int(problem_size.C));

  stride_output[0] = (int(problem_size.K));
  stride_output[1] = (int(problem_size.Q) * int(problem_size.K));
  stride_output[2] =
      (int(problem_size.Q) * int(problem_size.P) * int(problem_size.K));
  conv::SplitKMode split_k_mode = conv::SplitKMode::kSerial;
  Conv2dConfiguration configuration{split_k_mode,       problem_size,
                                    stride_activations, stride_filters,
                                    stride_output,      bias_c};
  ConvArguments arguments{
      ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta, get_scalar_pointer_mode()};
  if (tune) {
    Operation const* best_operation = operations[0];
    int best_split_k = 2;
    long long best_time = -1;
    int split_ks[] = {1, 2, 4};
    for (int split_k_id = 0; split_k_id < 3; split_k_id++) {
      int split_k = split_ks[split_k_id];
      configuration.problem_size.split_k_slices = split_k;
      for (auto& operation : operations) {
        // Query host work space size
        uint64_t host_workspace_size_needed =
            operation->get_host_workspace_size(&configuration);

        if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
          continue;
        }

        char host_workspace[kHostWorkspaceSize];

        // Query device workspace size
        uint64_t device_workspace_size_needed =
            operation->get_device_workspace_size(&configuration, &arguments);

        if (uint64_t(get_workspace_size()) < device_workspace_size_needed) {
          continue;
        }

        // Initialize host and device workspaces
        Status status = operation->initialize(&configuration, host_workspace,
                                              get_workspace(), get_stream());

        if (status != cutlass::Status::kSuccess) {
          std::cout << "cutlass_handle: 1155";
          continue;
        }
        status = operation->can_implement(&configuration, &arguments);
        if (status != cutlass::Status::kSuccess) {
          std::cout << "cutlass_handle: 1060";
          continue;
        }
        // Run the operator
        status = operation->run(&arguments, host_workspace, get_workspace(),
                                get_stream());

        if (status != cutlass::Status::kSuccess) {
          std::cout << "cutlass_handle: 1068";
          continue;
        }
        cudaStreamSynchronize(get_stream());
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start));
        for (int i = 0; i < times; ++i) {
          operation->run(&arguments, host_workspace, get_workspace(),
                         get_stream());
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float msec = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
        float time = 1000 * msec / times;
        ConvDescription const& desc =
            static_cast<ConvDescription const&>(operation->description());
        auto threadblock_shape = desc.tile_description.threadblock_shape;
        if (verbose_) {
          std::cout << "time: " << (float)time << " us"
                    << " Config:"
                    << " " << threadblock_shape.m() << " "
                    << threadblock_shape.n() << " " << threadblock_shape.k()
                    << " " << desc.tile_description.threadblock_stages << " "
                    << split_k << " " << desc.name << std::endl;
        }

        if (time > 0 && (best_time < 0 || time < best_time)) {
          best_time = time;
          best_operation = operation;
          best_split_k = split_k;
        }
      }  // end for operation
    }    // end for split_k
    if (best_time == -1) {
      return cutlass::Status::kErrorNotSupported;
    }
    TuneConfig best_config;
    ConvDescription const& desc =
        static_cast<ConvDescription const&>(best_operation->description());
    auto threadblock_shape = desc.tile_description.threadblock_shape;
    best_config.cta_m = threadblock_shape.m();
    best_config.cta_n = threadblock_shape.n();
    best_config.cta_k = threadblock_shape.k();
    best_config.stage = desc.tile_description.threadblock_stages;
    best_config.split_k = best_split_k;
    best_config.name = desc.name;
    *tune_config = best_config;
    if (verbose_) {
      std::cout << "best_config " << best_config.cta_m << " "
                << best_config.cta_n << " " << best_config.cta_k << " "
                << best_config.stage << " " << best_config.split_k << " "
                << best_config.name << " " << std::endl;
      std::cout << "N=" << N << " H=" << H << " W=" << W << " C=" << C
                << " K=" << K << " R=" << R << " S=" << S
                << " stride_h=" << stride_h << " stride_w=" << stride_h
                << std::endl;
      std::cout << "best time " << (float)best_time << " us" << std::endl;
    }
    last_operation_ = best_operation;

    // run the same setting in cudnn library
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto cudnn_data_type = DtypeConvert(element_A);
    auto cudnn_compute_type =
        (element_A == NumericTypeID::kS8) ? CUDNN_DATA_INT32 : CUDNN_DATA_FLOAT;

    cudnnTensorDescriptor_t cudnnIdesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&cudnnIdesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(cudnnIdesc, CUDNN_TENSOR_NHWC,
                                          cudnn_data_type, N, C, H, W));

    cudnnFilterDescriptor_t cudnnFdesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&cudnnFdesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(cudnnFdesc, cudnn_data_type,
                                          CUDNN_TENSOR_NHWC, K, C, R, S));

    cudnnTensorDescriptor_t cudnnOdesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&cudnnOdesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(cudnnOdesc, CUDNN_TENSOR_NHWC,
                                          cudnn_data_type, N, K, P, Q));

    cudnnConvolutionDescriptor_t cudnnConvDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        cudnnConvDesc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        CUDNN_CONVOLUTION, cudnn_compute_type));

    checkCUDNN(
        cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH));

    size_t space_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &space_size);
    void* workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    cudnnStatus_t status = cudnnConvolutionForward(
        handle, alpha, cudnnIdesc, ptr_A, cudnnFdesc, ptr_B, cudnnConvDesc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspace, space_size,
        beta, cudnnOdesc, ptr_D);

    if (status != CUDNN_STATUS_SUCCESS) {
      BLADNN_LOG(WARNING) << "CUDNN UNSUPPORTED CONV SHAPE";
      return cutlass::Status::kSuccess;
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < times; ++i) {
      cudnnConvolutionForward(handle, alpha, cudnnIdesc, ptr_A, cudnnFdesc,
                              ptr_B, cudnnConvDesc,
                              CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                              workspace, space_size, beta, cudnnOdesc, ptr_D);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float msec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    float cudnn_time = 1000 * msec / times;
    auto end = std::chrono::steady_clock::now();
    auto tuning_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           end - tuning_begin)
                           .count();
    if (verbose_) {
      std::cout << "cudnn time " << (float)cudnn_time << " us" << std::endl;
      std::cout << "tuning time " << tuning_time / 1000 << " ms" << std::endl;
    }
    if (cudnn_time < best_time) {
      return cutlass::Status::kErrorNotSupported;
    }

    if (best_time < 0) {
      return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kSuccess;
  }  // end if tune

  // replay
  configuration.problem_size.split_k_slices = tune_config->split_k;
  for (auto& operation : operations) {
    // Query host work space size
    uint64_t host_workspace_size_needed =
        operation->get_host_workspace_size(&configuration);

    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
      continue;
    }

    char host_workspace[kHostWorkspaceSize];

    // Query device workspace size
    uint64_t device_workspace_size_needed =
        operation->get_device_workspace_size(&configuration, &arguments);

    if (uint64_t(get_workspace_size()) < device_workspace_size_needed) {
      continue;
    }

    // Initialize host and device workspaces
    Status status = operation->initialize(&configuration, host_workspace,
                                          get_workspace(), get_stream());

    if (status != cutlass::Status::kSuccess) {
      continue;
    }

    status = operation->can_implement(&configuration, &arguments);
    if (status != cutlass::Status::kSuccess) {
      continue;
    }

    // Run the operator
    // cudaStreamSynchronize(get_stream());
    // auto begin = std::chrono::steady_clock::now();
    // for (int i = 0; i < ITERS; ++i) {
    status = operation->run(&arguments, host_workspace, get_workspace(),
                            get_stream());
    // }
    // cudaStreamSynchronize(get_stream());
    // auto end = std::chrono::steady_clock::now();
    // auto time =
    //     std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
    //         .count();
    // std::cout << "replay time " << (float)time / ITERS << " us" << std::endl;
    if (status == cutlass::Status::kSuccess) {
      return status;
    }
  }
  return cutlass::Status::kErrorNotSupported;
}
}  // namespace library
}  // namespace cutlass
