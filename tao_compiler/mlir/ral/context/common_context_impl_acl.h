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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_ACL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_ACL_H_

#if defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)

// The majority part of this file is copied from the corresponding
// impelmentation of arm compute library, we re-implement these layers to add
// thread safety.

#include <memory>

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IMemoryRegion.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/ral_base.h"
#include "mlir/ral/ral_helper.h"
#include "src/core/helpers/MemoryHelpers.h"

namespace arm_compute {

// Forward declarations
class ITensor;
class NEDepthwiseConvolutionLayerNativeKernel;

/** Basic function to simulate a convolution layer. This function calls one of
 * the following functions:
 * -# @ref cpu::CpuGemm     (executed only in case GEMM is required for the
 * operation)
 * -# @ref cpu::CpuWinogradConv2d (executed only in case Winograd is required
 * for the operation)
 * -# @ref cpu::CpuDirectConv2d   (executed only in case Direct Convolution is
 * required for the operation)
 * -# @ref NEFFTConvolutionLayer      (executed only in case FFT is required for
 * the operation)
 *
 *
 * The function selects one of the algorithms mentioned above based on:
 *      - The size of the kernel
 *      - Number of input/output feature maps
 *      - Amount of memory needed
 *
 * Generally GEMM-based convolution is executed when neither Winograd nor FFT
 * nor Direct convolution can be performed.
 *
 * FP32 Algorithm| Filter Size                                        |
 * Input/Output feature maps               |
 * --------------|----------------------------------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5(fast maths) 7x1 1x7        |  Input
 * channels is greater than 3         | FFT           | Squared kernels and
 * greater than 9x9               |  Input feature maps > Output feature maps |
 * DirectConv    | 9x9                                                | | GEMM
 * | Any size                                           | |
 *
 * Winograd 5x5 requires fast maths enabled.
 *
 * FP16 Algorithm| Filter Size      |
 * --------------|------------------|
 * Winograd      | Not supported    |
 * FFT           | Not supported    |
 * DirectConv    | 9x9              |
 * GEMM          | Any size         |
 *
 *
 */
class DISCNEConvolutionLayer : public IFunction {
 public:
  /** Constructor */
  DISCNEConvolutionLayer(
      std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEConvolutionLayer(const DISCNEConvolutionLayer&) = delete;
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEConvolutionLayer& operator=(const DISCNEConvolutionLayer&) = delete;
  /** Default move constructor */
  DISCNEConvolutionLayer(DISCNEConvolutionLayer&&) = default;
  /** Prevent instances of this class from being moved (As this class contains
   * non movable objects) */
  DISCNEConvolutionLayer& operator=(DISCNEConvolutionLayer&&) = default;
  /** Default destructor */
  ~DISCNEConvolutionLayer();
  /** Set the input and output tensors.
   *
   * Valid data layouts:
   * - NHWC
   * - NCHW
   *
   * Valid data type configurations:
   * |src0           |src1               |src2   |dst            |
   * |:--------------|:------------------|:------|:--------------|
   * |F16            |F16                |F16    |F16            |
   * |F32            |F32                |F32    |F32            |
   * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
   * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
   * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
   * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
   *
   * @param[in]  input            Source tensor. 3 lower dimensions represent a
   * single input [width, height, IFM], while every optional dimension from 4
   * and above represent a batch of inputs. Data types supported:
   * QASYMM8/QASYMM8_SIGNED/F16/F32.
   * @param[in]  weights          Weights tensor. Weights are 4D tensor with
   * dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p
   * input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
   * @param[in]  biases           Biases tensor. Shared biases supported. Biases
   * are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input,
   * except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of
   * S32 type.
   * @param[out] output           Destination tensor. 3 lower dimensions
   * represent a single output [width, height, OFM], while the rest represent
   * batch of outputs. Data types supported: Same as @p input.
   * @param[in]  conv_info        Contains padding and stride information
   * described in @ref PadStrideInfo.
   * @param[in]  weights_info     Specifies if the weights tensor has been
   * reshaped with NEWeightsReshapeKernel. If this is not part of the fully
   * connected layer the weights tensor has also been transposed with
   * cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p
   * input.
   * @param[in]  dilation         (Optional) Dilation, in elements, across x and
   * y. Defaults to (1, 1).
   * @param[in]  act_info         (Optional) Activation layer information in
   * case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU
   * supported.
   * @param[in]  enable_fast_math (Optional) Enable fast math computation. In
   * case this flag were set, the function could dispatch the fastest
   * implementation available which may introduce a drop of accuracy as well.
   * Default is false
   * @param[in]  num_groups       (Optional) Number of groups when performing a
   * grouped convolution. num_groups != 1 is not supported
   */
  void configure(ITensor* input, const ITensor* weights, const ITensor* biases,
                 ITensor* output, const PadStrideInfo& conv_info,
                 const WeightsInfo& weights_info = WeightsInfo(),
                 const Size2D& dilation = Size2D(1U, 1U),
                 const ActivationLayerInfo& act_info = ActivationLayerInfo(),
                 bool enable_fast_math = false, unsigned int num_groups = 1);
  /** Static function to check if given info will lead to a valid configuration
   * of @ref NEConvolutionLayer
   *
   * @param[in] input            Source tensor. 3 lower dimensions represent a
   * single input [width, height, IFM], while every optional dimension from 4
   * and above represent a batch of inputs. Data types supported:
   * QASYMM8/QASYMM8_SIGNED/F16/F32.
   * @param[in] weights          Weights tensor. Weights are 4D tensor with
   * dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p
   * input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
   * @param[in] biases           Biases tensor. Shared biases supported. Biases
   * are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input,
   * except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of
   * S32 type.
   * @param[in] output           Destination tensor. 3 lower dimensions
   * represent a single output [width, height, OFM], while the rest represent
   * batch of outputs. Data types supported: Same as @p input.
   * @param[in] conv_info        Contains padding and stride information
   * described in @ref PadStrideInfo.
   * @param[in] weights_info     Specifies if the weights tensor has been
   * reshaped with NEWeightsReshapeKernel. If this is not part of the fully
   * connected layer the weights tensor has also been transposed with
   * cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p
   * input.
   * @param[in] dilation         (Optional) Dilation, in elements, across x and
   * y. Defaults to (1, 1).
   * @param[in] act_info         (Optional) Activation layer information in case
   * of a fused activation.
   * @param[in] enable_fast_math (Optional) Enable fast math computation. In
   * case this flag were set, the function could dispatch the fastest
   * implementation available which may introduce a drop of accuracy as well.
   * Default is false
   * @param[in] num_groups       (Optional) Number of groups when performing a
   * grouped convolution. num_groups != 1 is not supported
   *
   * @return a status
   */
  static Status validate(
      const ITensorInfo* input, const ITensorInfo* weights,
      const ITensorInfo* biases, const ITensorInfo* output,
      const PadStrideInfo& conv_info,
      const WeightsInfo& weights_info = WeightsInfo(),
      const Size2D& dilation = Size2D(1U, 1U),
      const ActivationLayerInfo& act_info = ActivationLayerInfo(),
      bool enable_fast_math = false, unsigned int num_groups = 1);
  /** Static function to check if given info will return the convolution called
   * by @ref NEConvolutionLayer
   *
   * @param[in] input            Source tensor. 3 lower dimensions represent a
   * single input [width, height, IFM], while every optional dimension from 4
   * and above represent a batch of inputs. Data types supported:
   * QASYMM8/QASYMM8_SIGNED/F16/F32.
   * @param[in] weights          Weights tensor. Weights are 4D tensor with
   * dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p
   * input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
   * @param[in] output           Destination tensor. 3 lower dimensions
   * represent a single output [width, height, OFM], while the rest represent
   * batch of outputs. Data types supported: Same as @p input.
   * @param[in] conv_info        Contains padding and stride information
   * described in @ref PadStrideInfo.
   * @param[in] weights_info     Specifies if the weights tensor has been
   * reshaped with NEWeightsReshapeKernel. If this is not part of the fully
   * connected layer the weights tensor has also been transposed with
   * cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p
   * input.
   * @param[in] dilation         (Optional) Dilation, in elements, across x and
   * y. Defaults to (1, 1).
   * @param[in] act_info         (Optional) Activation layer information in case
   * of a fused activation.
   * @param[in] enable_fast_math (Optional) Enable fast math computation. In
   * case this flag were set, the function could dispatch the fastest
   * implementation available which may introduce a drop of accuracy as well.
   * Default is false
   *
   * @return the Convolution Method Hint
   */
  static ConvolutionMethod get_convolution_method(
      const ITensorInfo* input, const ITensorInfo* weights,
      const ITensorInfo* output, const PadStrideInfo& conv_info,
      const WeightsInfo& weights_info = WeightsInfo(),
      const Size2D& dilation = Size2D(1U, 1U),
      const ActivationLayerInfo& act_info = ActivationLayerInfo(),
      bool enable_fast_math = false);
  // Inherited methods overridden:
  void run() override;
  void prepare() override;

  void run(ITensor* input, const ITensor* weights, const ITensor* biases,
           ITensor* output);
  void prepare(ITensor* input, const ITensor* weights, const ITensor* biases,
               ITensor* output);

  const ITensorPack& get_packed_weight();
  void reuse_packed_weight(const ITensorPack& pack);
  std::string get_md5_for_packed_weight();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/** Function to execute a depthwise convolution.
 */
class DISCNEDepthwiseConvolutionLayer : public IFunction {
 public:
  /** Default constructor */
  DISCNEDepthwiseConvolutionLayer(
      std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEDepthwiseConvolutionLayer(const DISCNEDepthwiseConvolutionLayer&) =
      delete;
  /** Default move constructor */
  DISCNEDepthwiseConvolutionLayer(DISCNEDepthwiseConvolutionLayer&&) = default;
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEDepthwiseConvolutionLayer& operator=(
      const DISCNEDepthwiseConvolutionLayer&) = delete;
  /** Default move assignment operator */
  DISCNEDepthwiseConvolutionLayer& operator=(
      DISCNEDepthwiseConvolutionLayer&&) = default;
  /** Default destructor */
  ~DISCNEDepthwiseConvolutionLayer();
  /** Initialize the function's source, destination, weights and convolution
   * information.
   *
   * Valid data layouts:
   * - NHWC
   * - NCHW
   *
   * Valid data type configurations:
   * |src0           |src1               |src2   |dst            |
   * |:--------------|:------------------|:------|:--------------|
   * |F16            |F16                |F16    |F16            |
   * |F32            |F32                |F32    |F32            |
   * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
   * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
   * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
   * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
   *
   * @param[in, out] input            Source tensor. Data type supported:
   * QASYMM8/QASYMM8_SIGNED/F16/F32
   * @param[out]     output           Destination tensor. Data type supported:
   * same as @p input.
   * @param[in]      weights          Weights tensor. These are 3D tensors with
   * shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input or
   * QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is
   * QASYMM8/QASYMM8_SIGNED.
   * @param[in]      biases           Biases tensor. A 1D tensor with shape
   * [IFM]. Must be nullptr if not needed. Data type supported: Same as @p
   * input, S32 when input is QASYMM8/QASYMM8_SIGNED.
   * @param[in]      conv_info        Padding and stride information to use for
   * the convolution.
   * @param[in]      depth_multiplier (Optional) Multiplier to apply to the
   * input's depth in order to retrieve the output's depth. Defaults to 1.
   * @param[in]      act_info         (Optional) Activation layer information in
   * case of a fused activation.
   * @param[in]      dilation         (Optional) Dilation, in elements, across x
   * and y. Defaults to (1, 1).
   */
  void configure(ITensor* input, const ITensor* weights, const ITensor* biases,
                 ITensor* output, const PadStrideInfo& conv_info,
                 unsigned int depth_multiplier = 1,
                 const ActivationLayerInfo& act_info = ActivationLayerInfo(),
                 const Size2D& dilation = Size2D(1U, 1U));

  /** Static function to check if given info will lead to a valid configuration
   * of @ref NEDepthwiseConvolutionLayer
   *
   * @param[in] input            Source tensor. Data type supported:
   * QASYMM8/QASYMM8_SIGNED/F16/F32
   * @param[in] output           Destination tensor. Data type supported: same
   * as @p input.
   * @param[in] weights          Weights tensor. These are 3D tensors with shape
   * [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input or
   * QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is
   * QASYMM8/QASYMM8_SIGNED.
   * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM].
   * Must be nullptr if not needed. Data type supported: Same as @p input, S32
   * when input is QASYMM8/QASYMM8_SIGNED.
   * @param[in] conv_info        Padding and stride information to use for the
   * convolution.
   * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's
   * depth in order to retrieve the output's depth. Defaults to 1.
   * @param[in] act_info         (Optional) Activation layer information in case
   * of a fused activation.
   * @param[in] dilation         (Optional) Dilation, in elements, across x and
   * y. Defaults to (1, 1).
   *
   * @return a status
   */
  static Status validate(
      const ITensorInfo* input, const ITensorInfo* weights,
      const ITensorInfo* biases, const ITensorInfo* output,
      const PadStrideInfo& conv_info, unsigned int depth_multiplier = 1,
      const ActivationLayerInfo& act_info = ActivationLayerInfo(),
      const Size2D& dilation = Size2D(1U, 1U));

  // Inherited methods overriden:
  void run() override;
  void prepare() override;

  void run(ITensor* input, const ITensor* weights, const ITensor* biases,
           ITensor* output);
  void prepare(ITensor* input, const ITensor* weights, const ITensor* biases,
               ITensor* output);

  const ITensorPack& get_packed_weight();
  void reuse_packed_weight(const ITensorPack& pack);
  std::string get_md5_for_packed_weight();

 private:
  /** Basic function to execute optimized depthwise convolution routines. This
   * function calls the following kernels:
   *
   * @note At the moment 3x3 and 5x5 convolution of stride 1, 2 are supported
   *
   * -# @ref NEFillBorderKernel (if pad_x or pad_y > 0) and no assembly kernel
   * implementation is present
   * -# @ref NEDepthwiseConvolutionLayer3x3Kernel if 3x3 and no assembly kernel
   * implementation is present
   * -# @ref cpu::CpuDepthwiseConvolutionAssemblyDispatch if assembly kernel
   * implementation is present
   * -# @ref NEDirectConvolutionLayerOutputStageKernel if re-quantization of
   * output is required
   * -# @ref NEActivationLayer if fused activation is required
   *
   */
  class NEDepthwiseConvolutionLayerOptimizedInternal : public IFunction {
   public:
    /** Default constructor */
    NEDepthwiseConvolutionLayerOptimizedInternal(
        std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class
     * contains pointers) */
    NEDepthwiseConvolutionLayerOptimizedInternal(
        const NEDepthwiseConvolutionLayerOptimizedInternal&) = delete;
    /** Default move constructor */
    NEDepthwiseConvolutionLayerOptimizedInternal(
        NEDepthwiseConvolutionLayerOptimizedInternal&&) = default;
    /** Prevent instances of this class from being copied (As this class
     * contains pointers) */
    NEDepthwiseConvolutionLayerOptimizedInternal& operator=(
        const NEDepthwiseConvolutionLayerOptimizedInternal&) = delete;
    /** Default move assignment operator */
    NEDepthwiseConvolutionLayerOptimizedInternal& operator=(
        NEDepthwiseConvolutionLayerOptimizedInternal&&) = default;
    /** Default destructor */
    ~NEDepthwiseConvolutionLayerOptimizedInternal() = default;
    /** Initialize the function's source, destination, kernels and border_size.
     *
     * @param[in, out] input            Source tensor. Data type supported:
     * QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
     * @param[in]      weights          Weights tensor. These are 3D tensors
     * with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p
     * input.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape
     * [IFM]. Must be nullptr if not needed. Data type supported: Same as @p
     * input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out]     output           Destination tensor. Data type supported:
     * same as @p input.
     * @param[in]      conv_info        Padding and stride information to use
     * for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the
     * input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information
     * in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across
     * x and y. Defaults to (1, 1).
     */
    void configure(ITensor* input, const ITensor* weights,
                   const ITensor* biases, ITensor* output,
                   const PadStrideInfo& conv_info,
                   unsigned int depth_multiplier = 1,
                   const ActivationLayerInfo& act_info = ActivationLayerInfo(),
                   const Size2D& dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid
     * configuration of @ref NEDepthwiseConvolutionLayer3x3
     *
     * @param[in] input            Source tensor. Data type supported:
     * QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
     * @param[in] weights          Weights tensor. These are 3D tensors with
     * shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
     * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM].
     * Must be nullptr if not needed. Data type supported: Same as @p input, S32
     * when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output           Destination tensor. Data type supported: same
     * as @p input.
     * @param[in] conv_info        Padding and stride information to use for the
     * convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's
     * depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in
     * case of a fused activation.
     * @param[in] dilation         (Optional) Dilation, in elements, across x
     * and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(
        const ITensorInfo* input, const ITensorInfo* weights,
        const ITensorInfo* biases, const ITensorInfo* output,
        const PadStrideInfo& conv_info, unsigned int depth_multiplier = 1,
        const ActivationLayerInfo& act_info = ActivationLayerInfo(),
        const Size2D& dilation = Size2D(1U, 1U));

    // Inherited methods overriden:
    void run() override;
    void prepare() override;

    void run(ITensor* input, const ITensor* weights, const ITensor* biases,
             ITensor* output);
    void prepare(ITensor* input, const ITensor* weights, const ITensor* biases,
                 ITensor* output);

    const ITensorPack& get_packed_weight();
    void reuse_packed_weight(const ITensorPack& pack);
    std::string get_md5_for_packed_weight();

   private:
    MemoryGroup _memory_group;
    struct Impl;
    std::unique_ptr<Impl> _impl;
  };

  /** Basic function to execute a generic depthwise convolution. This function
   * calls the following kernel:
   *
   * -# @ref NEDepthwiseConvolutionLayerNativeKernel
   *
   */
  class NEDepthwiseConvolutionLayerGeneric : public IFunction {
   public:
    /** Default constructor */
    NEDepthwiseConvolutionLayerGeneric();
    /** Prevent instances of this class from being copied (As this class
     * contains pointers) */
    NEDepthwiseConvolutionLayerGeneric(
        const NEDepthwiseConvolutionLayerGeneric&) = delete;
    /** Default move constructor */
    NEDepthwiseConvolutionLayerGeneric(NEDepthwiseConvolutionLayerGeneric&&) =
        default;
    /** Prevent instances of this class from being copied (As this class
     * contains pointers) */
    NEDepthwiseConvolutionLayerGeneric& operator=(
        const NEDepthwiseConvolutionLayerGeneric&) = delete;
    /** Default move assignment operator */
    NEDepthwiseConvolutionLayerGeneric& operator=(
        NEDepthwiseConvolutionLayerGeneric&&) = default;
    /** Default destructor */
    ~NEDepthwiseConvolutionLayerGeneric() = default;
    /** Initialize the function's source, destination, weights and convolution
     * information.
     *
     * @param[in, out] input            Source tensor. Data type supported:
     * QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
     * @param[out]     output           Destination tensor. Data type supported:
     * same as @p input.
     * @param[in]      weights          Weights tensor. These are 3D tensors
     * with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p
     * input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is
     * QASYMM8/QASYMM8_SIGNED.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape
     * [IFM]. Must be nullptr if not needed. Data type supported: Same as @p
     * input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]      conv_info        Padding and stride information to use
     * for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the
     * input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information
     * in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across
     * x and y. Defaults to (1, 1).
     */
    void configure(ITensor* input, const ITensor* weights,
                   const ITensor* biases, ITensor* output,
                   const PadStrideInfo& conv_info,
                   unsigned int depth_multiplier = 1,
                   const ActivationLayerInfo& act_info = ActivationLayerInfo(),
                   const Size2D& dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid
     * configuration of @ref NEDepthwiseConvolutionLayerGeneric
     *
     * @param[in] input            Source tensor. Data type supported:
     * QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
     * @param[in] output           Destination tensor. Data type supported: same
     * as @p input.
     * @param[in] weights          Weights tensor. These are 3D tensors with
     * shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input or
     * QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is
     * QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM].
     * Must be nullptr if not needed. Data type supported: Same as @p input, S32
     * when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] conv_info        Padding and stride information to use for the
     * convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's
     * depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in
     * case of a fused activation.
     * @param[in] dilation         (Optional) Dilation, in elements, across x
     * and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(
        const ITensorInfo* input, const ITensorInfo* weights,
        const ITensorInfo* biases, const ITensorInfo* output,
        const PadStrideInfo& conv_info, unsigned int depth_multiplier = 1,
        const ActivationLayerInfo& act_info = ActivationLayerInfo(),
        const Size2D& dilation = Size2D(1U, 1U));

    // Inherited methods overriden:
    void run() override;

    void run(ITensor* input, const ITensor* weights, const ITensor* biases,
             ITensor* output);
    void prepare(ITensor* input, const ITensor* weights, const ITensor* biases,
                 ITensor* output);

    const ITensorPack& get_packed_weight();
    void reuse_packed_weight(const ITensorPack& pack);
    std::string get_md5_for_packed_weight();

   private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
  };
  MemoryGroup _memory_group;
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

class DISCNEGEMMLowpMatrixMultiplyCore : public IFunction {
 public:
  /** Constructor */
  DISCNEGEMMLowpMatrixMultiplyCore(
      std::shared_ptr<IMemoryManager> memory_manager = nullptr,
      IWeightsManager* weights_manager = nullptr);
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEGEMMLowpMatrixMultiplyCore(const DISCNEGEMMLowpMatrixMultiplyCore&) =
      delete;
  /** Default move constructor */
  DISCNEGEMMLowpMatrixMultiplyCore(DISCNEGEMMLowpMatrixMultiplyCore&&) =
      default;
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCNEGEMMLowpMatrixMultiplyCore& operator=(
      const DISCNEGEMMLowpMatrixMultiplyCore&) = delete;
  /** Default move assignment operator */
  DISCNEGEMMLowpMatrixMultiplyCore& operator=(
      DISCNEGEMMLowpMatrixMultiplyCore&&) = default;
  /** Default destructor */
  ~DISCNEGEMMLowpMatrixMultiplyCore();
  /** Initialise the kernel's inputs, output
   *
   * Valid data layouts:
   * - NHWC
   * - NCHW
   *
   * Valid data type configurations:
   * |src0           |src1               |src2     |dst            |
   * |:--------------|:------------------|:--------|:--------------|
   * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
   * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
   * |QASYMM8        |QSYMM8             |S32      |QASYMM8        |
   * |QASYMM8        |QASYMM8            |S32      |S32            |
   * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |S32            |
   * |QASYMM8        |QSYMM8             |S32      |S32            |
   * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
   * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
   * |QASYMM8_SIGNED |QSYMM8             |S32      |QASYMM8_SIGNED |
   * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |S32            |
   * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |S32            |
   * |QASYMM8_SIGNED |QSYMM8             |S32      |S32            |
   *
   * @note GEMM_LOWP:  low precision GEMM kernel
   *  This kernel performs the following computations:
   *
   *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of
   * them.
   *  -# Convert b values from QASYMM8 to int32 add b_offset to each of them.
   *  -# Compute the matrix product of the resulting a * b in int32.
   *
   * @note The @p output type is S32 if @p gemm_info.type ==
   * GEMMLowpOutputStageType::NONE. It is QASYMM8/QASYMM8_SIGNED otherwise
   *
   * @param[in]  a         First input tensor  (Matrix A). Data type supported:
   * QASYMM8/QASYMM8_SIGNED.
   * @param[in]  b         Second input tensor (Matrix B). Data type supported:
   * QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL.
   * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr.
   * Data type supported: S32
   * @param[out] output    Output tensor. Data type supported: Data type
   * supported: S32/QASYMM8/QASYMM8_SIGNED
   * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B
   * have been reshaped and if the reshape of matrix B should be executed only
   * for the first run
   */
  void configure(const ITensor* a, const ITensor* b, const ITensor* c,
                 ITensor* output, const GEMMInfo& gemm_info = GEMMInfo());
  /** Static function to check if given info will lead to a valid configuration
   * of @ref NEGEMMLowpMatrixMultiplyCore
   *
   * Similar to @ref NEGEMMLowpMatrixMultiplyCore::configure()
   *
   * @return a status
   */
  static Status validate(const ITensorInfo* a, const ITensorInfo* b,
                         const ITensorInfo* c, const ITensorInfo* output,
                         const GEMMInfo& gemm_info = GEMMInfo());

  // Inherited methods overridden
  void run() override;
  void prepare() override;

  void run(ITensor* a, const ITensor* b, const ITensor* c, ITensor* output);
  void prepare(ITensor* a, const ITensor* b, const ITensor* c, ITensor* output);

  const ITensorPack& get_packed_weight();
  void reuse_packed_weight(const ITensorPack& pack);
  std::string get_md5_for_packed_weight();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace arm_compute

#endif  // defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_ACL_H_
