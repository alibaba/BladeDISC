/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination with a maximum operation used by
  epilogues.
*/

#pragma once

#include <cutlass/half.h>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int Count,                ///< Number of elements computed per operation
                ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                ///< but we use 64 or 32 sometimes when there are not enough
                ///< data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default,  ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationRelu_Scale {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentScaleBias = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = false;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;             ///< scales accumulators
    ElementCompute beta;              ///< scales source tensor
    ElementCompute threshold;         ///< minimum value that is output
    ElementCompute const* alpha_ptr;  ///< pointer to accumulator scalar - if
                                      ///< not null, loads it from memory
    ElementCompute const* beta_ptr;   ///< pointer to source scalar - if not
                                      ///< null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          threshold(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta = ElementCompute(0),
           ElementCompute threshold = ElementCompute(0))
        : alpha(alpha),
          beta(beta),
          threshold(threshold),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr,
           ElementCompute const* beta_ptr = nullptr,
           ElementCompute threshold = ElementCompute(0))
        : alpha(0),
          beta(0),
          threshold(threshold),
          alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementCompute threshold_;

 public:
  bool source_needed = true;
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  LinearCombinationRelu_Scale(Params const& params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return source_needed; }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementCompute const&>(allones);
    }
  }

  /// Computes per-channel linear scaling and bias : D = scale * accumulator +
  /// bias Scale and Bias are from input Fragment
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator,
                            FragmentScaleBias const& scale,
                            FragmentScaleBias const& bias) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform per-channel scale and bias
    FragmentCompute intermediate;

    multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate = mul_add_accumulator(scale, converted_accumulator,
                                       bias);  // D = scale * Accum + bias

    ReLu<FragmentCompute> relu;

    // Compute threshold optionally
    intermediate = relu(threshold_, intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes per-channel linear scaling and bias : D = scale * accumulator +
  /// bias Scale and Bias are from input Fragment
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator,
                            FragmentOutput const& source,
                            FragmentScaleBias const& scale,
                            FragmentScaleBias const& bias) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_source = source_converter(source);

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform per-channel scale and bias
    FragmentCompute intermediate;

    multiply_add<FragmentCompute> mul_add_accumulator;
    // multiplies<FragmentCompute> mul_accumulator;
    multiplies<FragmentCompute> mul_add_source;

    intermediate = mul_add_source(alpha_, converted_source);
    intermediate =
        mul_add_accumulator(scale, converted_accumulator, intermediate);

    ReLu<FragmentCompute> relu;

    // Compute threshold optionally
    intermediate = relu(threshold_, intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
