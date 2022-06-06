//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeUtils.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/IR/PatternMatch.h"          // from @llvm-project
#include "llvm/Support/FormatVariadic.h"
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

namespace mlir {
namespace mhlo {

/*
// Common function for lowering reduce operations to MHLO ops.
template <typename T>
llvm::Optional<Value> convertReduceOpCommon(
    PatternRewriter &rewriter, Operation *op, RankedTensorType output_type,
    Value input_value, ElementsAttr axes_elems, bool keep_dims,
    Type reduce_element_type, bool is_quantized, double input_scale,
    int64_t input_zp, double output_scale, int64_t output_zp) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  ArrayRef<int64_t> input_shape = input_type.getShape();
  ArrayRef<int64_t> output_shape = output_type.getShape();
  auto input_rank = input_shape.size();
  Value val = input_value;

  if (axes_elems.getNumElements() == 0) {
    // // No axes means return the original tensor.
    // auto identity_op = CreateOpAndInfer<mhlo::IdentityOp>(
    //     rewriter, op->getLoc(), output_type, val);
    // val = identity_op.getResult();
  } else {
    // Reduce along each axis
    SmallVector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    for (int i = 0; i < axes_elems.getNumElements(); i++) {
      int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
      if (axis_val < 0)
        axis_val += input_rank;
      auto axis_attr = rewriter.getI64IntegerAttr(axis_val);

      shape_vec[axis_val] = 1;
      RankedTensorType reduce_type =
          RankedTensorType::get(shape_vec, reduce_element_type);

      auto reduce_op = CreateOpAndInfer<T>(rewriter, op->getLoc(), reduce_type,
                                           val, axis_attr);

      val = reduce_op.getResult();
    }
    // Optionally squeeze out the reduced axes.
    if (!keep_dims) {
      auto reshape_op = CreateOpAndInfer<mhlo::ReshapeOp>(
          rewriter, op->getLoc(), output_type, val,
          rewriter.getI64ArrayAttr(output_shape));
      val = reshape_op.getResult();
    }
  }

  return val;
}

// Lowers ReduceAll to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceAllOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  return convertReduceOpCommon<mhlo::ReduceAllOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceAny to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceAnyOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  return convertReduceOpCommon<mhlo::ReduceAnyOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMin to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceMinOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  return convertReduceOpCommon<mhlo::ReduceMinOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMax to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceMaxOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  return convertReduceOpCommon<mhlo::ReduceMaxOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceProd to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceProdOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  return convertReduceOpCommon<mhlo::ReduceProdOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceSum to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceSumOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();
  return convertReduceOpCommon<mhlo::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      reduce_element_type, input_is_qtype, input_scale, input_zp, output_scale,
      output_zp);
}

// Lowers ReduceMean to a sequence of MHLO ops.
llvm::Optional<Value>
convertReduceMeanOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims) {
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type)
    return llvm::None;

  int64_t input_rank = input_type.getRank();
  int64_t num_elems_on_reduced_axis = 1;
  for (int i = 0; i < axes_elems.getNumElements(); i++) {
    int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
    if (axis_val < 0)
      axis_val += input_rank;
    num_elems_on_reduced_axis *= input_type.getShape()[axis_val];
  }
  double div_scale = 1.0 / static_cast<double>(num_elems_on_reduced_axis);

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();

  auto val = convertReduceOpCommon<mhlo::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, keep_dims,
      reduce_element_type, input_is_qtype, input_scale, input_zp, output_scale,
      output_zp);

  if (!val.hasValue())
    return llvm::None;

  if (!input_is_qtype) {
    Value div_const = getMhloConstTensorSingleF32(rewriter, op, div_scale);
    return CreateOpAndInfer<mhlo::MulOp>(rewriter, op->getLoc(), output_type,
                                         val.getValue(), div_const, 0)
        .getResult();
  }

  return val;
}*/

} // namespace mhlo
} // namespace mlir
