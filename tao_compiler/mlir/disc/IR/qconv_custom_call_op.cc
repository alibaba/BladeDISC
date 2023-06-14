/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file defines quantized conv related custom calls.

#include "lhlo/IR/lhlo_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/disc/IR/custom_call_base.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace mhlo_disc {

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value MaybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  if (!type.isIndex() && !value.getType().isIndex()) {
    // in case of i32 -> i64 or vice versa
    Value casted = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
    return b.create<arith::IndexCastOp>(loc, type, casted);
  }
  return b.create<arith::IndexCastOp>(loc, type, value);
}

LogicalResult reifyReturnTypeShapesDconvI8I8I8Impl(
    CustomCallOp op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return op->emitError() << "mismatch #operands or #results\n";

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value padding = op->getOperand(2);

  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
  auto paddingTy = padding.getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy || !paddingTy)
    return op->emitError() << "not support unranked type\n";

  Location loc = op.getLoc();
  Type shape_scalar_type = paddingTy.getElementType();
  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, loc, v, shape_scalar_type);
  };

  SmallVector<Value> spatial_padding_values;
  auto config = op.getBackendConfig().cast<DictionaryAttr>();
  auto dimension_numbers =
      config.getAs<mhlo::ConvDimensionNumbersAttr>("dimension_numbers");
  auto input_spatial_dimensions_attr =
      dimension_numbers.getInputSpatialDimensions();
  int64_t padding_num = input_spatial_dimensions_attr.size() * 2;
  for (int64_t i = 0; i < padding_num; i++) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
    Value pad_value = to_shape_scalar_type(
        builder.create<tensor::ExtractOp>(loc, padding, offset));
    spatial_padding_values.push_back(pad_value);
  }

  int64_t input_batch_dimension = dimension_numbers.getInputBatchDimension();
  int64_t kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto kernel_spatial_dimensions_attr =
      dimension_numbers.getKernelSpatialDimensions();
  auto output_spatial_dimensions_attr =
      dimension_numbers.getOutputSpatialDimensions();

  SmallVector<Value, 4> shape_values(output_spatial_dimensions_attr.size() + 2);

  // batch dim = lhs-batch-dim / batch_group_count
  auto batch_group_count_attr = config.getAs<IntegerAttr>("batch_group_count");
  Value lhs_batch_dim = to_shape_scalar_type(
      builder.create<tensor::DimOp>(loc, lhs, input_batch_dimension));
  Value batch_group_count =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
          loc, batch_group_count_attr.getInt()));
  Value batch_dim = to_shape_scalar_type(
      builder.create<arith::DivSIOp>(loc, lhs_batch_dim, batch_group_count));
  int64_t output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  shape_values[output_batch_dimension] = batch_dim;

  // Output's feature dim is the same with kernel's output feature dim.
  Value feature_dim = to_shape_scalar_type(
      builder.create<tensor::DimOp>(loc, rhs, kernel_output_feature_dimension));
  int64_t output_feature_dimension =
      dimension_numbers.getOutputFeatureDimension();
  shape_values[output_feature_dimension] = feature_dim;

  auto window_strides_attr =
      config.getAs<DenseIntElementsAttr>("window_strides");
  auto rhs_dilation_attr = config.getAs<DenseIntElementsAttr>("rhs_dilation");

  Value one =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(loc, 1));
  for (uint64_t i = 0; i < output_spatial_dimensions_attr.size(); i++) {
    // effective_input_value =
    //    (input_size - 1) * input_dilation + 1 + padding_left + padding_right
    Value effective_input_value =
        to_shape_scalar_type(builder.create<tensor::DimOp>(
            loc, lhs, input_spatial_dimensions_attr[i]));

    // Padding.
    if (!spatial_padding_values.empty()) {
      Value padding_left = spatial_padding_values[i * 2];
      Value padding_right = spatial_padding_values[i * 2 + 1];
      effective_input_value = builder.create<arith::AddIOp>(
          loc, effective_input_value,
          builder.create<arith::AddIOp>(loc, padding_left, padding_right));
    }

    // effective_kernel_size = (kernel_size - 1) * dilation + 1
    Value effective_kernel_size_value =
        to_shape_scalar_type(builder.create<tensor::DimOp>(
            loc, rhs, kernel_spatial_dimensions_attr[i]));
    Value kernel_dilation =
        to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
            loc, rhs_dilation_attr.getValues<int64_t>()[i]));
    effective_kernel_size_value = builder.create<arith::AddIOp>(
        loc, one,
        builder.create<arith::MulIOp>(
            loc, kernel_dilation,
            builder.create<arith::SubIOp>(loc, effective_kernel_size_value,
                                          one)));

    // output_size =
    //     (effective_input_value - effective_kernel_size_value) / stride + 1
    Value output_dim_value = builder.create<arith::SubIOp>(
        loc, effective_input_value, effective_kernel_size_value);
    if (window_strides_attr) {
      Value stride_value =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, window_strides_attr.getValues<int64_t>()[i]));
      output_dim_value =
          builder.create<arith::DivSIOp>(loc, output_dim_value, stride_value);
    }
    output_dim_value =
        builder.create<arith::AddIOp>(loc, output_dim_value, one);
    shape_values[output_spatial_dimensions_attr[i]] = output_dim_value;
  }
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, shape_values);
  reifiedReturnShapes.push_back(output_shape);
  return success();
}

}  // namespace mhlo_disc
namespace lmhlo_disc {

Value getRootMemRef(Value memref) {
  Value rootMemRef = memref;
  while (Operation* operandOp = rootMemRef.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(operandOp))
      break;
    rootMemRef = operandOp->getOperand(0);
  }
  return rootMemRef;
}

Value GetConvMetadata(Operation* op, PatternRewriter& rewriter, int rank) {
  // Metadata:
  //   - input layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - kernel layout: each field for one dimension. The order is:
  //     * in_channel, out_channel, spatial dimensions
  //   - output layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - strides: each filed for one spatial dimension.
  //   - dilations: each filed for one spatial dimension.
  //   - weight_is_const : indicate whether the weight is const.
  Location loc = op->getLoc();
  Type field_type = rewriter.getI32Type();
  int num_spatial_dims = rank - 2;
  int num_metadata_fields = rank * 3 + (rank - 2) * 2 + 1;
  Value metadata_value = rewriter.create<memref::AllocaOp>(
      loc, MemRefType::get(
               {num_metadata_fields}, field_type, MemRefLayoutAttrInterface(),
               StringAttr::get(op->getContext(), placement_utils::kCpu)));
  std::vector<int64_t> fields;
  auto config = op->getAttrOfType<DictionaryAttr>("backend_config");
  auto dimension_numbers =
      config.getAs<mhlo::ConvDimensionNumbersAttr>("dimension_numbers");
  // input layout
  fields.push_back(dimension_numbers.getInputBatchDimension());
  fields.push_back(dimension_numbers.getInputFeatureDimension());
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  fields.insert(fields.end(), input_spatial_dimensions.begin(),
                input_spatial_dimensions.end());
  // kernel layout
  fields.push_back(dimension_numbers.getKernelInputFeatureDimension());
  fields.push_back(dimension_numbers.getKernelOutputFeatureDimension());
  auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  fields.insert(fields.end(), kernel_spatial_dimensions.begin(),
                kernel_spatial_dimensions.end());
  // output layout
  fields.push_back(dimension_numbers.getOutputBatchDimension());
  fields.push_back(dimension_numbers.getOutputFeatureDimension());
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  fields.insert(fields.end(), output_spatial_dimensions.begin(),
                output_spatial_dimensions.end());
  // strides
  auto window_strides_attr =
      config.getAs<DenseIntElementsAttr>("window_strides");
  auto window_strides = disc_ral::ConvertDenseIntAttr(window_strides_attr);
  fields.insert(fields.end(), window_strides.begin(), window_strides.end());
  // rhs_dilation
  auto rhs_dilation_attr = config.getAs<DenseIntElementsAttr>("rhs_dilation");
  auto rhs_dilation = disc_ral::ConvertDenseIntAttr(rhs_dilation_attr);
  fields.insert(fields.end(), rhs_dilation.begin(), rhs_dilation.end());
  fields.push_back(disc_ral::isConstantMemRef(op->getOperand(1)));

  for (auto&& en : llvm::enumerate(fields)) {
    Value value =
        rewriter.create<arith::ConstantIntOp>(loc, en.value(), field_type);
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, en.index());
    SmallVector<Value, 1> ivs(1, offset);
    rewriter.create<memref::StoreOp>(loc, value, metadata_value, ivs);
  }

  return metadata_value;
}

LogicalResult lowerToLibraryCallDconvI8I8I8Impl(CustomCallOp op,
                                                PatternRewriter& rewriter,
                                                Value ctx,
                                                Value stream_handle) {
  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return op->emitError() << "mismatch #operands or #results\n";
  SmallVector<Value, 4> newOperands{stream_handle};

  // input
  newOperands.push_back(op.getOperand(0));
  // kernel
  newOperands.push_back(op.getOperand(1));
  // padding
  newOperands.push_back(op.getOperand(2));
  // input scales
  newOperands.push_back(op.getOperand(3));
  // filter scales
  newOperands.push_back(op.getOperand(4));
  // output scales
  newOperands.push_back(op.getOperand(5));
  // output
  newOperands.push_back(op.getOperand(6));
  // input & kernel & output layouts, strides and dilation
  auto inputTy = newOperands[1].getType().cast<MemRefType>();
  newOperands.push_back(GetConvMetadata(op, rewriter, inputTy.getRank()));

  bool on_gpu = placement_utils::isGpuMemRef(op->getOperand(2));
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, ctx, newOperands, "ral_qconv_s8_s8_s8", false,
      on_gpu ? "gpu" : "cpu");
  return success();
}

}  // namespace lmhlo_disc

REGISTER_CUSTOM_CALL("ral_qconv_s8_s8_s8",
                     mhlo_disc::reifyReturnTypeShapesDconvI8I8I8Impl,
                     lmhlo_disc::lowerToLibraryCallDconvI8I8I8Impl);

}  // namespace mlir
