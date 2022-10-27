#include <cstdint>
#include <iostream>

#include "bladnn/backend/cutlass/gemm/per_channel_quant/device/gemm.h"
#include "bladnn/backend/cutlass/epilogue/linear_combination_relu.h"
#include "bladnn/bladnn.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/host_tensor.h"

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

using ElementAccumulator = int32_t;    // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = int8_t;  // <- data type of elements in input matrix A
using ElementInputB = int8_t;  // <- data type of elements in input matrix B
using ElementOutput = int8_t;  // <- data type of elements in output matrix D
// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N =
                                             // 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 64,
                             64>;  // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu_Scale<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::
              value,     // <- the number of elements per vectorized
                         // memory access. For a byte, it's 16
                         // elements. This becomes the vector width of
                         // math instructions in the epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::
        OnlyAlphaPerChannelScaling>;  // <- data type for alpha/beta in linear
                                      // combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm_Scale<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

namespace bladnn {
bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, int8_t* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          int8_t* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, int8_t* d_ptr,
          int c_dim0, int c_dim1, int batch_count, bool a_is_const,
          bool b_is_const, float* alpha, float* beta, float* scale, float* bias,
          int8_t* c_ptr) {
  if (a_dtype != Dtype::kS8 || b_dtype != Dtype::kS8 || c_dtype != Dtype::kS8 ||
      scale == nullptr || bias == nullptr) {
    return false;
  }
  int m = a_transpose ? a_dim1 : a_dim0;
  int k = a_transpose ? a_dim0 : a_dim1;
  int n = b_transpose ? b_dim0 : b_dim1;
  int lda = a_dim1;
  int ldb = b_dim1;
  int ldc = c_dim1;

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_device_a(a_ptr, lda);
  cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_device_b(b_ptr, ldb);
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_device_c(c_ptr, ldc);
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_device_d(d_ptr, ldc);
  cutlass::TensorRef<ElementComputeEpilogue, LayoutInputA> tensor_device_scale(
      scale, n);
  cutlass::TensorRef<ElementComputeEpilogue, LayoutInputA> tensor_device_bias(
      bias, n);

  ElementComputeEpilogue alpha_val = ElementComputeEpilogue(1);
  if (alpha == nullptr) {
    alpha = &alpha_val;
  }
  ElementComputeEpilogue beta_val = ElementComputeEpilogue(0);
  if (beta == nullptr) {
    beta = &beta_val;
  }

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel

  typename Gemm::Arguments arguments{
      problem_size,     // <- problem size of matrix multiplication
      tensor_device_a,  // <- reference to matrix A on device
      tensor_device_b,  // <- reference to matrix B on device
      tensor_device_scale, tensor_device_bias,
      tensor_device_c,  // <- reference to matrix C on device
      tensor_device_d,  // <- reference to matrix D on device
      {*alpha, *beta},  // <- tuple of alpha and beta
      split_k_slices};  // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
  return true;
}
}  // namespace bladnn