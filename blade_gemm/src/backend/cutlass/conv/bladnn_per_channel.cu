#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "bladnn/backend/cutlass/conv/per_channel_quant/device/implicit_gemm_convolution.h"
#include "bladnn/backend/cutlass/epilogue/linear_combination_relu.h"
#include "bladnn/backend/cutlass/conv/per_channel_quant/kernel/default_conv2d_fprop.h"
#include "bladnn/bladnn.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
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

using ElementAccumulator = int32_t;  // Data type of accumulator
using ElementComputeEpilogue =
    float;  // Data type of epilogue computation (alpha, beta)
using ElementInputA = int8_t;  // Data type of elements in input tensor
using ElementInputB = int8_t;  // Data type of elements in input tensor
using ElementOutput = int8_t;  // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using LayoutScale = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ThreadblockShape =
    cutlass::gemm::GemmShape<128, 128, 128>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;  // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape =
    cutlass::gemm::GemmShape<8, 8, 16>;  // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// This code section describes the epilogue part of the kernel, we use default
// value
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu_Scale<
    ElementOutput,  // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::
              value,     // The number of elements per vectorized.
                         // memory access. This becomes the vector width of
                         // math instructions in the epilogue too.
    ElementAccumulator,  // Data type of accumulator
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::
        OnlyAlphaPerChannelScaling>;  // Data type for alpha/beta in linear
                                      // combination

using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultConv2dFprop_Scale<
        ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
        LayoutOutput, ElementAccumulator, MMAOp, SmArch, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOp, SwizzleThreadBlock, NumStages,
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution_Scale<Conv2dFpropKernel>;

namespace bladnn {
bool conv2d(Context* ctx, Dtype in_dtype, Dtype out_dtype, ConvKind conv_kind,
            Layout data_layout, Layout kernel_layout, int N, int H, int W,
            int C, int K, int R, int S, int P, int Q, int pad_h, int pad_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int groups, int8_t* ptr_A, int8_t* ptr_B, int8_t* ptr_D,
            float* alpha, float* beta, float* scale, float* bias,
            int8_t* ptr_C) {
  if (in_dtype != Dtype::kS8 || out_dtype != Dtype::kS8) {
    return false;
  }

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);

  cutlass::Tensor4DCoord padding(1, pad_h, pad_w, 1);
  cutlass::MatrixCoord conv_stride(stride_h, stride_w);
  cutlass::MatrixCoord dilation(dilation_h, dilation_w);

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride, dilation, output_size,
      mode, split_k_slices);

  cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_device_a(
      ptr_A, LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_device_b(
      ptr_B, LayoutInputB::packed(filter_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_device_c(
      ptr_C, LayoutOutput::packed(output_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_device_d(
      ptr_D, LayoutOutput::packed(output_size));

  cutlass::TensorRef<ElementComputeEpilogue, LayoutScale> tensor_device_scale(
      scale, K);
  cutlass::TensorRef<ElementComputeEpilogue, LayoutScale> tensor_device_bias(
      bias, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha_val = ElementComputeEpilogue(1);
  if (alpha == nullptr) {
    alpha = &alpha_val;
  }
  ElementComputeEpilogue beta_val = ElementComputeEpilogue(0);
  if (beta == nullptr) {
    beta = &beta_val;
  }

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel

  typename ImplicitGemm::Arguments arguments{
      problem_size,       tensor_device_a, tensor_device_b, tensor_device_scale,
      tensor_device_bias, tensor_device_c, tensor_device_d, {*alpha, *beta},
  };

  ImplicitGemm implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  //
  // Launch initialized CUTLASS kernel
  //
  status = implicit_gemm_op();

  CUTLASS_CHECK(status);
  return true;
}
}  // namespace bladnn