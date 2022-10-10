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

// This file defines the operations used in the DISC RAL dialect.

#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

#include <mutex>
#include <unordered_map>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "tensorflow/compiler/mlir/disc/IR/custom_call_base.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_enums.cc.inc"

namespace mlir {
namespace mhlo_disc {

using llvm::StringRef;

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// mhlo disc Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDiscDialect::MhloDiscDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDiscDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.cc.inc"

      >();
  context->loadDialect<tensor::TensorDialect>();
}

//===----------------------------------------------------------------------===//
// H2DOp
//===----------------------------------------------------------------------===//

LogicalResult H2DOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                           operands[0], &reifiedReturnShapes);
}

LogicalResult H2DOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// D2HOp
//===----------------------------------------------------------------------===//

LogicalResult D2HOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                           operands[0], &reifiedReturnShapes);
}

LogicalResult D2HOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  CustomCallOp::Adaptor adaptor(operands);
  ValueRange args = adaptor.args();
  StringRef target = call_target_name();
  auto reify_shapes_func =
      CustomCallRegistry::Global().FindReifyShapesFunc(target.str());
  if (!reify_shapes_func) {
    return emitOpError() << "custom call " << target << " is not supported.";
  }
  return reify_shapes_func(*this, builder, operands, reifiedReturnShapes);
}

LogicalResult CustomCallOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// FakeQuantOp
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult QuantVerify(T* op) {
  auto inputTy = op->input().getType().template dyn_cast<RankedTensorType>();
  auto scaleTy = op->scale().getType().template dyn_cast<RankedTensorType>();
  auto zeroPointTy =
      op->zero_point().getType().template dyn_cast<RankedTensorType>();
  auto resultTy = op->result().getType().template dyn_cast<RankedTensorType>();
  if (!inputTy || !scaleTy || !zeroPointTy || !resultTy)
    return op->emitOpError() << "only support ranked input.\n";
  if (inputTy.getShape() != resultTy.getShape())
    return op->emitOpError() << "input and result have mismatch shape.\n";
  if (scaleTy.getRank() != zeroPointTy.getRank())
    return op->emitOpError() << "scale and zero_point have mismatch rank.\n";
  auto axis = op->axis().template getValues<int64_t>();
  if (axis.size() != scaleTy.getRank())
    return op->emitOpError() << "num of quantized axes (len(axis)) is not "
                                "equal to the rank of scale tensor\n";
  return success();
}

LogicalResult FakeQuantOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizeOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// QuantizedDotGeneralOp
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult CommonVerifyForQuantizedComputeIntensiveOp(T* op) {
  auto inputTy = op->input().getType().template dyn_cast<RankedTensorType>();
  auto weightTy = op->weight().getType().template dyn_cast<RankedTensorType>();
  auto resultTy = op->result().getType().template dyn_cast<RankedTensorType>();

  if (!inputTy || !weightTy || !resultTy ||
      inputTy.getRank() != weightTy.getRank() ||
      inputTy.getRank() != resultTy.getRank()) {
    return op->emitOpError()
           << "input, weight and result should have the same rank.\n";
  }

  auto inputScaleTy =
      op->input_scale().getType().template dyn_cast<RankedTensorType>();
  auto inputZeroPointTy =
      op->input_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!inputScaleTy || !inputZeroPointTy || inputScaleTy.getRank() != 0 ||
      inputZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "input_scale and input_zero_point only support "
                                "per-tensor quantization\n";
  }

  auto resultScaleTy =
      op->result_scale().getType().template dyn_cast<RankedTensorType>();
  auto resultZeroPointTy =
      op->result_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!resultScaleTy || !resultZeroPointTy || resultScaleTy.getRank() != 0 ||
      resultZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "result_scale and result_zero_point only "
                                "support per-tensor quantization\n";
  }

  auto weightScaleTy =
      op->weight_scale().getType().template dyn_cast<RankedTensorType>();
  auto weightZeroPointTy =
      op->weight_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!weightScaleTy || !weightZeroPointTy ||
      weightScaleTy.getShape() != weightZeroPointTy.getShape()) {
    return op->emitOpError()
           << "weight_scale and weight_zero_point have mismatch shape\n";
  }
  auto axis = op->axis().template getValues<int64_t>();
  if (axis.size() != weightScaleTy.getRank())
    return op->emitOpError() << "num of quantized axes (len(axis)) is not "
                                "equal to the rank of weight_scale tensor\n";
  return success();
}

LogicalResult QuantizedDotGeneralOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

LogicalResult QuantizedDotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = input().getType().dyn_cast<ShapedType>();
  auto rhsType = weight().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dimNumbers = dot_dimension_numbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.input(), lhsDim));
  }

  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.input(), i));
    }
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.weight(), i));
    }
  }
  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// QuantizedDynamicConvOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizedDynamicConvOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

//===----------------------------------------------------------------------===//
// Mostly copy from the implementation of mhlo::DynamicConvOp
//===----------------------------------------------------------------------===//

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  if (!type.isIndex() && !value.getType().isIndex()) {
    // in case of i32 -> i64 or vice versa
    Value casted = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
    return b.create<arith::IndexCastOp>(loc, type, casted);
  }
  return b.create<arith::IndexCastOp>(loc, type, value);
}

template <typename Op>
LogicalResult ConvReifyReturnTypeImpl(
    Op* op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes,
    const SmallVector<Value>& spatial_padding_values, Type shape_scalar_type) {
  typename Op::Adaptor adaptor(operands);
  Value lhs = adaptor.input();
  Value rhs = adaptor.weight();

  RankedTensorType lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
  RankedTensorType rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!lhs_type || !rhs_type) return failure();

  Location loc = op->getLoc();

  auto to_shape_scalar_type = [&](Value v) {
    return maybeCastTo(builder, loc, v, shape_scalar_type);
  };

  auto dimension_numbers = op->dimension_numbers();
  int64_t input_batch_dimension = dimension_numbers.getInputBatchDimension();
  int64_t kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto input_spatial_dimensions_attr =
      dimension_numbers.getInputSpatialDimensions();
  auto kernel_spatial_dimensions_attr =
      dimension_numbers.getKernelSpatialDimensions();
  auto output_spatial_dimensions_attr =
      dimension_numbers.getOutputSpatialDimensions();

  SmallVector<Value, 4> shape_values(output_spatial_dimensions_attr.size() + 2);

  // batch dim = lhs-batch-dim / batch_group_count
  Value lhs_batch_dim = to_shape_scalar_type(
      builder.create<tensor::DimOp>(loc, lhs, input_batch_dimension));
  Value batch_group_count = to_shape_scalar_type(
      builder.create<arith::ConstantIndexOp>(loc, op->batch_group_count()));
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

  Optional<DenseIntElementsAttr> window_strides_attr = op->window_strides();
  Optional<DenseIntElementsAttr> lhs_dilation_attr = op->lhs_dilation();
  Optional<DenseIntElementsAttr> rhs_dilation_attr = op->rhs_dilation();

  Value one =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(loc, 1));
  for (uint64_t i = 0; i < output_spatial_dimensions_attr.size(); i++) {
    // effective_input_value =
    //    (input_size - 1) * input_dilation + 1 + padding_left + padding_right
    Value effective_input_value =
        to_shape_scalar_type(builder.create<tensor::DimOp>(
            loc, lhs, input_spatial_dimensions_attr[i]));
    // Dilation.
    if (lhs_dilation_attr) {
      Value input_dilation =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, lhs_dilation_attr.getValue().getValues<int64_t>()[i]));
      effective_input_value = builder.create<arith::AddIOp>(
          loc,
          builder.create<arith::MulIOp>(
              loc,
              builder.create<arith::SubIOp>(loc, effective_input_value, one),
              input_dilation),
          one);
    }

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
    if (rhs_dilation_attr) {
      Value kernel_dilation =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, rhs_dilation_attr.getValue().getValues<int64_t>()[i]));
      effective_kernel_size_value = builder.create<arith::AddIOp>(
          loc, one,
          builder.create<arith::MulIOp>(
              loc, kernel_dilation,
              builder.create<arith::SubIOp>(loc, effective_kernel_size_value,
                                            one)));
    }

    // output_size =
    //     (effective_input_value - effective_kernel_size_value) / stride + 1
    Value output_dim_value = builder.create<arith::SubIOp>(
        loc, effective_input_value, effective_kernel_size_value);
    if (window_strides_attr) {
      Value stride_value =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, window_strides_attr.getValue().getValues<int64_t>()[i]));
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

LogicalResult QuantizedDynamicConvOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  QuantizedDynamicConvOp::Adaptor adaptor(operands);
  Value d_padding = adaptor.d_padding();

  RankedTensorType padding_type =
      d_padding.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!padding_type) return failure();

  Location loc = this->getLoc();
  Type shape_scalar_type = padding_type.getElementType();
  auto to_shape_scalar_type = [&](Value v) {
    return maybeCastTo(builder, loc, v, shape_scalar_type);
  };

  SmallVector<Value> spatial_padding_values;
  auto dimension_numbers = this->dimension_numbers();
  auto input_spatial_dimensions_attr =
      dimension_numbers.getInputSpatialDimensions();
  int64_t padding_num = input_spatial_dimensions_attr.size() * 2;
  for (int64_t i = 0; i < padding_num; i++) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
    Value pad_value = to_shape_scalar_type(
        builder.create<tensor::ExtractOp>(loc, d_padding, offset));
    spatial_padding_values.push_back(pad_value);
  }

  return ConvReifyReturnTypeImpl(this, builder, operands, reifiedReturnShapes,
                                 spatial_padding_values, shape_scalar_type);
}

//===----------------------------------------------------------------------===//
// SparseReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult SparseReshapeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseReshapeOp::Adaptor adaptor(operands);
  auto input_indices_type =
      adaptor.input_indices().getType().dyn_cast<RankedTensorType>();
  auto input_shape_type =
      adaptor.input_shape().getType().dyn_cast<RankedTensorType>();
  auto new_shape_type =
      adaptor.new_shape().getType().dyn_cast<RankedTensorType>();
  if (!input_indices_type || !input_shape_type || !new_shape_type) {
    return failure();
  }

  if (input_indices_type.getRank() != 2 || input_shape_type.getRank() != 1) {
    return failure();
  }
  Location loc = this->getLoc();

  Value num_values = builder.create<tensor::DimOp>(loc, operands[0], 0);
  Value new_rank = builder.create<tensor::DimOp>(loc, operands[2], 0);
  // output indices
  SmallVector<Value, 2> output_indices_shape_values;
  output_indices_shape_values.push_back(num_values);
  output_indices_shape_values.push_back(new_rank);
  Value output_indices_shape =
      builder.create<tensor::FromElementsOp>(loc, output_indices_shape_values);
  reifiedReturnShapes.push_back(output_indices_shape);

  // output shape
  SmallVector<Value, 1> output_shape_shape_values;
  Value output_rank =
      builder.create<tensor::DimOp>(loc, operands[operands.size() - 1], 0);
  output_shape_shape_values.push_back(output_rank);
  Value output_shape_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_shape_values);
  reifiedReturnShapes.push_back(output_shape_shape);
  return success();
}

LogicalResult SparseReshapeOp::verify() {
  auto input_indices_type =
      this->input_indices().getType().template dyn_cast<RankedTensorType>();
  auto input_shape_type =
      this->input_shape().getType().template dyn_cast<RankedTensorType>();
  auto new_shape_type =
      this->new_shape().getType().template dyn_cast<RankedTensorType>();
  auto output_indices_type =
      this->output_indices().getType().template dyn_cast<RankedTensorType>();
  auto output_shape_type =
      this->output_shape().getType().template dyn_cast<RankedTensorType>();

  if (!input_indices_type || !input_shape_type || !new_shape_type) {
    return this->emitOpError() << "only support ranked input.\n";
  }
  if (!input_indices_type.getElementType().isInteger(64) ||
      !input_shape_type.getElementType().isInteger(64) ||
      !new_shape_type.getElementType().isInteger(64)) {
    return this->emitOpError() << "only support int64 input.\n";
  }
  if (input_indices_type.getRank() != 2 || output_indices_type.getRank() != 2) {
    return this->emitOpError() << "Input/Output indices must be a matrix.\n";
  }
  if (input_shape_type.getRank() != 1 || new_shape_type.getRank() != 1 ||
      output_shape_type.getRank() != 1) {
    return this->emitOpError()
           << "Input/Output shape and new shape must be a vector.\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseFillEmptyRowsOp
//===----------------------------------------------------------------------===//

LogicalResult SparseFillEmptyRowsOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseFillEmptyRowsOp::Adaptor adaptor(operands);
  // index 0
  auto indices_type = adaptor.indices().getType().cast<RankedTensorType>();
  // index 1
  auto values_type = adaptor.values().getType().cast<RankedTensorType>();
  // index 2
  auto dense_shape_type =
      adaptor.dense_shape().getType().cast<RankedTensorType>();
  // index 3
  auto default_value_type =
      adaptor.default_value().getType().cast<RankedTensorType>();

  Location loc = this->getLoc();

  Value num_indices = builder.create<tensor::DimOp>(loc, operands[0], 0);
  Value rank = builder.create<tensor::DimOp>(loc, operands[0], 1);
  Value idx_zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // TODO(lanbo.llb): Handle corner case when dense_rows == 0
  Value dense_rows = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(),
      builder.create<tensor::ExtractOp>(loc, operands[2], idx_zero));
  // outputs:
  // 0: output_indices
  // 1: output_values
  // 2: empty_row_indicator
  // 3: reverse_index_map
  // 4: output_elements(scalar)

  // However N_full can only be determined by the values from indices
  // so we allocate a output_indices/output_values to max possible size, which
  // is `N + dense_rows`
  Value N_full = builder.create<arith::AddIOp>(loc, num_indices, dense_rows);

  // output indices: {N + dense_rows, rank}
  SmallVector<Value, 2> output_indices_shape_values;
  output_indices_shape_values.push_back(N_full);
  output_indices_shape_values.push_back(rank);
  Value output_indices_shape =
      builder.create<tensor::FromElementsOp>(loc, output_indices_shape_values);
  reifiedReturnShapes.push_back(output_indices_shape);

  // output values: {N + dense_rows}
  SmallVector<Value, 2> output_values_shape_values;
  output_values_shape_values.push_back(N_full);
  Value output_values_shape =
      builder.create<tensor::FromElementsOp>(loc, output_values_shape_values);
  reifiedReturnShapes.push_back(output_values_shape);

  // empty indicator: {dense_rows}
  SmallVector<Value, 1> empty_row_indicator_shape_values;
  empty_row_indicator_shape_values.push_back(dense_rows);
  Value empty_row_indicator_shape = builder.create<tensor::FromElementsOp>(
      loc, empty_row_indicator_shape_values);
  reifiedReturnShapes.push_back(empty_row_indicator_shape);

  // reverse index map: {N}
  SmallVector<Value, 1> reverse_index_map_shape_values;
  reverse_index_map_shape_values.push_back(num_indices);
  Value reverse_index_map_shape = builder.create<tensor::FromElementsOp>(
      loc, reverse_index_map_shape_values);
  reifiedReturnShapes.push_back(reverse_index_map_shape);

  // output_elements
  Value idx_one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 1> output_elements_shape_values;
  output_elements_shape_values.push_back(idx_one);
  Value output_elements_shape =
      builder.create<tensor::FromElementsOp>(loc, output_elements_shape_values);
  reifiedReturnShapes.push_back(output_elements_shape);

  return success();
}

LogicalResult SparseFillEmptyRowsOp::verify() {
  auto indices_type = this->indices().getType().dyn_cast<RankedTensorType>();
  auto values_type = this->values().getType().dyn_cast<RankedTensorType>();
  auto dense_shape_type =
      this->dense_shape().getType().dyn_cast<RankedTensorType>();
  auto default_value_type =
      this->default_value().getType().dyn_cast<RankedTensorType>();

  if (!indices_type || !values_type || !default_value_type) {
    return failure();
  }

  if (indices_type.getRank() != 2) {
    return this->emitOpError() << "indices must be a matrix";
  }
  if (values_type.getRank() != 1) {
    return this->emitOpError() << "values must be a vector";
  }
  if (default_value_type.getRank() != 0) {
    return this->emitOpError() << "default_value must be a scalar";
  }
  if (!dense_shape_type.hasStaticShape()) {
    return this->emitOpError() << "DISC only support static-rank optimization, "
                                  "thus dense_shape should has static shape";
  }

  if (dense_shape_type.getRank() != 1) {
    return this->emitOpError() << "dense_shape must be a vector";
  }

  auto output_indices_type =
      this->output_indices().getType().dyn_cast<RankedTensorType>();
  auto output_values_type =
      this->output_values().getType().dyn_cast<RankedTensorType>();
  auto empty_row_indicator_type =
      this->empty_row_indicator().getType().dyn_cast<RankedTensorType>();
  auto reverse_index_map_type =
      this->reverse_index_map().getType().dyn_cast<RankedTensorType>();
  auto output_elements_type =
      this->output_elements().getType().dyn_cast<RankedTensorType>();
  if (!output_indices_type || !output_values_type ||
      !empty_row_indicator_type || !reverse_index_map_type ||
      !output_elements_type) {
    return failure();
  }

  if (output_indices_type.getRank() != 2) {
    return this->emitOpError() << "output_indices must be a matrix";
  }

  if (output_values_type.getRank() != 1 ||
      empty_row_indicator_type.getRank() != 1 ||
      reverse_index_map_type.getRank() != 1 ||
      output_elements_type.getRank() != 1) {
    return this->emitOpError()
           << "outputs 1-4 for mhlo_disc::sparse_fill_empty_rows must be a "
              "vector";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseSegmentMeanOp
//===----------------------------------------------------------------------===//

LogicalResult SparseSegmentMeanOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseSegmentMeanOp::Adaptor adaptor(operands);
  Location loc = this->getLoc();
  auto data_type = adaptor.data().getType().cast<RankedTensorType>();
  auto indices_type = adaptor.indices().getType().cast<RankedTensorType>();
  auto segment_ids_type =
      adaptor.segment_ids().getType().cast<RankedTensorType>();
  auto input_rank = data_type.getRank();
  SmallVector<Value, 2> output_shape_values;
  Value segment_size = builder.create<tensor::DimOp>(loc, operands[2], 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value last_segment_index =
      builder.create<arith::SubIOp>(loc, segment_size, one);
  Value last_segment_id_plus_one = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(),
          builder.create<tensor::ExtractOp>(loc, operands[2],
                                            last_segment_index)),
      one);
  output_shape_values.push_back(last_segment_id_plus_one);
  for (auto i = 1; i < input_rank; i++) {
    output_shape_values.push_back(
        builder.create<tensor::DimOp>(loc, operands[0], i));
  }
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_values);
  reifiedReturnShapes.push_back(output_shape);
  return success();
}

LogicalResult SparseSegmentMeanOp::verify() {
  auto data_type = this->data().getType().dyn_cast<RankedTensorType>();
  auto indices_type = this->indices().getType().dyn_cast<RankedTensorType>();
  auto segment_ids_type =
      this->segment_ids().getType().dyn_cast<RankedTensorType>();

  if (!data_type || !indices_type || !segment_ids_type) {
    return failure();
  }

  if (indices_type.getRank() != 1) {
    return this->emitOpError() << "indices should be a vector";
  }
  if (segment_ids_type.getRank() != 1) {
    return this->emitOpError() << "segment_ids should be a vector";
  }
  // only check when has static shape
  if (indices_type.hasStaticShape() && segment_ids_type.hasStaticShape()) {
    auto num_indices = indices_type.getDimSize(0);
    auto num_segment_ids = segment_ids_type.getDimSize(0);
    if (num_indices != num_segment_ids) {
      return this->emitOpError()
             << "segment_ids and indices should have same size";
    }
  }

  if (data_type.getRank() < 1) {
    return this->emitOpError() << "input must be at least rank 1";
  }

  auto output_type = this->output().getType().dyn_cast<RankedTensorType>();
  if (!output_type) {
    return failure();
  }

  if (output_type.getRank() != data_type.getRank()) {
    return this->emitOpError() << "output must have the same rank as input";
  }

  return success();
}

}  // namespace mhlo_disc
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.cc.inc"
