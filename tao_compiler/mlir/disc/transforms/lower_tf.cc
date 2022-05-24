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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                          // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"           // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/rng_uniform_custom_call_op.h"
#include "tensorflow/compiler/mlir/disc/IR/topk_custom_call_op.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "disc-lower-tf"

namespace mlir {
namespace disc_ral {
namespace {

ValueRange PackRandomUniformInputs(Value lb, Value ub, Value shape) {
  return {lb, ub, shape};
}

StringAttr PackRandomUniformBackendConfig(IntegerAttr seed, IntegerAttr seed2,
                                          PatternRewriter* rewriter) {
  mhlo_disc::RngUniformBackendConfig config(seed.getValue().getSExtValue(),
                                            seed2.getValue().getSExtValue());
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << ::llvm::json::toJSON(config);
  return rewriter->getStringAttr(ostream.str());
}

// Prepare TF operations in functions for subsequent legalization.
struct PrepareTFPass : public DiscLowerTfPassBase<PrepareTFPass> {
  using DiscLowerTfPassBase<PrepareTFPass>::DiscLowerTfPassBase;

  // TODO: move to td file
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mhlo_disc::MhloDiscDialect>();
  }

  void runOnOperation() override;
};

// Converts a tf.SqueezeOp to xla_hlo.ReshapeOp
// SqueezeOp with empty squeeze_dims is a dynamic rank op and will not be
// supported
class ConvertSqueezeOpDynamic : public OpRewritePattern<TF::SqueezeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::SqueezeOp op,
                                PatternRewriter& rewriter) const final {
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getType(),
                                                   op.input());
    } else {
      Location loc = op.getLoc();
      Value input = op.input();
      auto input_ty = input.getType().dyn_cast<RankedTensorType>();
      if (!input_ty) {
        return failure();
      }
      int64_t input_rank = input_ty.getRank();
      auto squeeze_dims_attr = op.squeeze_dims();
      int64_t squeeze_dims_attr_size = squeeze_dims_attr.size();
      if (squeeze_dims_attr_size == 0) {
        return failure();
      }
      llvm::SetVector<int64_t> squeeze_dims;
      for (int64_t i = 0; i < squeeze_dims_attr_size; ++i) {
        squeeze_dims.insert(squeeze_dims_attr[i].cast<IntegerAttr>().getInt());
      }
      SmallVector<Value, 4> shape_values;
      int64_t output_rank = input_rank - squeeze_dims_attr_size;
      shape_values.reserve(output_rank);
      for (int64_t i = 0; i < input_rank; ++i) {
        if (squeeze_dims.count(i)) {
          continue;
        }
        auto dim_size = input_ty.getDimSize(i);
        if (dim_size == ShapedType::kDynamicSize) {
          shape_values.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
        } else {
          shape_values.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, dim_size));
        }
      }
      Value new_shape =
          rewriter.create<tensor::FromElementsOp>(loc, shape_values);
      rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
          op, result_ty, op.input(), new_shape);
    }
    return success();
  }
};

// Converts a tf.TopKV2Op to topk custom_call
class ConvertTopKV2OpDynamic : public OpRewritePattern<TF::TopKV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::TopKV2Op op,
                                PatternRewriter& rewriter) const final {
    auto k_type = op.k().getType().dyn_cast<RankedTensorType>();
    if (!k_type || (k_type.getElementType() != rewriter.getIntegerType(32)) ||
        (k_type.getRank() != 0)) {
      return op.emitOpError() << "TopKV2 requires 0D scalar tensor k";
    }

    auto input_type = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_type) {
      return op.emitOpError() << "TopKV2 requires input to be ranked tensor";
    }
    int64_t input_rank = input_type.getRank();
    int64_t last_dim_index = input_rank - 1;

    // Create an Itoa op for indices.
    // TODO(zk): if we always choose to use tf kernel implementation, then
    // this iota is redundant and should be removed.
    Type iota_type = RankedTensorType::get(input_type.getShape(),
                                           rewriter.getIntegerType(32));
    SmallVector<Value, 4> iota_shape_values;
    iota_shape_values.reserve(input_rank);
    for (int64_t idx = 0; idx < input_rank; ++idx) {
      Value dim = rewriter.create<tensor::DimOp>(op.getLoc(), op.input(), idx);
      iota_shape_values.push_back(dim);
    }
    Value iota_shape =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), iota_shape_values);
    Value iota_op = rewriter.create<mhlo::DynamicIotaOp>(
        op.getLoc(), iota_type, iota_shape,
        rewriter.getI64IntegerAttr(last_dim_index));

    // Create the topk custom call. It takes 3 inputs: keys, values and scalar
    // k. And generates two output: sorted_topk_keys, sorted_topk_values
    mhlo_disc::TopKBackendConfig backend_config(last_dim_index);
    std::string str;
    llvm::raw_string_ostream ostream(str);
    ostream << ::llvm::json::toJSON(backend_config);
    auto topk_custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
        op.getLoc(),
        TypeRange{op.getResult(0).getType(), op.getResult(1).getType()},
        ValueRange{op.input(), iota_op, op.k()},
        /*call_target_name*/ "topk",
        /*has_side_effect*/ false,
        /*backend_config*/ ostream.str());
    rewriter.replaceOp(op, {topk_custom_call_op.getResult(0),
                            topk_custom_call_op.getResult(1)});
    return success();
  }
};

// Convert a tf.RandomUniformOp to random_uniform custom_call
class ConvertUniformOp : public OpRewritePattern<TF::RandomUniformOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::RandomUniformOp op,
                                PatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    Value zero = rewriter.create<mhlo::ConstOp>(
        loc, rewriter.getFloatAttr(op.dtype(), 0.0));
    Value one = rewriter.create<mhlo::ConstOp>(
        loc, rewriter.getFloatAttr(op.dtype(), 1.0));
    auto cfg = PackRandomUniformBackendConfig(
        rewriter.getIntegerAttr(op.dtype(), op.seed()),
        rewriter.getIntegerAttr(op.dtype(), op.seed2()), &rewriter);
    auto custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
        loc, TypeRange{op.getResult().getType()},
        ValueRange{zero, one, op.shape()},
        /*custom call target*/ rewriter.getStringAttr("rng_uniform"),
        /*has_side_effect*/ rewriter.getBoolAttr(false),
        /*backend_config*/ cfg);
    rewriter.replaceOp(op, {custom_call_op.getResult(0)});
    return success();
  }
};

IntegerType QuantizedTypeToIntegerType(Type ty) {
  auto ctx = ty.getContext();
  if (ty.isa<TF::Qint8Type>()) {
    return IntegerType::get(ctx, 8, IntegerType::SignednessSemantics::Signed);
  } else if (ty.isa<TF::Qint16Type>()) {
    return IntegerType::get(ctx, 16, IntegerType::SignednessSemantics::Signed);
  } else if (ty.isa<TF::Qint32Type>()) {
    return IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signed);
  } else if (ty.isa<TF::Quint8Type>()) {
    return IntegerType::get(ctx, 8, IntegerType::SignednessSemantics::Unsigned);
  } else if (ty.isa<TF::Quint16Type>()) {
    return IntegerType::get(ctx, 16,
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return {};
}

struct NumericLimits {
  float lowest_quantized;
  float lower_bound_float;
  float upper_bound_float;
};

NumericLimits GetNumericLimits(IntegerType ty) {
  NumericLimits limits;
  if (ty.isSigned()) {
    if (ty.getWidth() == 8) {
      limits.lowest_quantized = std::numeric_limits<int8_t>::min();
      limits.lower_bound_float = std::numeric_limits<int8_t>::min();
      limits.upper_bound_float = std::numeric_limits<int8_t>::max();
    } else if (ty.getWidth() == 16) {
      limits.lowest_quantized = std::numeric_limits<int16_t>::min();
      limits.lower_bound_float = std::numeric_limits<int16_t>::min();
      limits.upper_bound_float = std::numeric_limits<int16_t>::max();
    } else {
      assert(ty.getWidth() == 32);
      limits.lowest_quantized = std::numeric_limits<int32_t>::min();
      limits.lower_bound_float = std::numeric_limits<int32_t>::min();
      limits.lower_bound_float =
          std::max(limits.lower_bound_float, -2.147483648e+09f);
      limits.upper_bound_float = std::numeric_limits<int32_t>::max();
      limits.upper_bound_float =
          std::min(limits.upper_bound_float, +2.147483520e+09f);
    }
  } else {
    assert(ty.isUnsigned());
    if (ty.getWidth() == 8) {
      limits.lowest_quantized = std::numeric_limits<uint8_t>::min();
      limits.lower_bound_float = std::numeric_limits<uint8_t>::min();
      limits.upper_bound_float = std::numeric_limits<uint8_t>::max();
    } else {
      assert(ty.getWidth() == 16);
      limits.lowest_quantized = std::numeric_limits<uint16_t>::min();
      limits.lower_bound_float = std::numeric_limits<uint16_t>::min();
      limits.upper_bound_float = std::numeric_limits<uint16_t>::max();
    }
  }
  return limits;
}

struct ConvertQuantizeV2Op : public OpRewritePattern<TF::QuantizeV2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::QuantizeV2Op op,
                                PatternRewriter& rewriter) const final {
    auto inputTy = op.input().getType().dyn_cast<RankedTensorType>();
    auto inputMinRangeTy =
        op.min_range().getType().dyn_cast<RankedTensorType>();
    auto inputMaxRangeTy =
        op.max_range().getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.output().getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !inputMinRangeTy || !inputMaxRangeTy || !resultTy)
      return failure();

    auto resultElemTy = QuantizedTypeToIntegerType(resultTy.getElementType());
    if (!resultElemTy) return failure();

    // TODO(disc): support other float type
    if (!inputTy.getElementType().isF32() ||
        !inputMinRangeTy.getElementType().isF32() ||
        !inputMaxRangeTy.getElementType().isF32()) {
      return failure();
    }

    // Not supported according to:
    //   tensorflow/core/kernels/quantize_op.cc
    if (op.axis() != -1 && op.mode() == "MIN_FIRST") return failure();

    // TODO(disc): support `MIN_COMBINED` mode
    if (op.mode() == "MIN_COMBINED") return failure();

    // TODO(disc): support `HALF_TO_EVEN` mode
    if (op.round_mode() == "HALF_TO_EVEN") return failure();

    if (op.axis() == -1 && op.mode() == "MIN_FIRST") {
      return quantizePerElement(op, rewriter);
    }

    return quantizePerChannel(op, rewriter);
  }

  LogicalResult adjustMinMaxRange(TF::QuantizeV2Op op,
                                  PatternRewriter& rewriter, Value& minRange,
                                  Value& maxRange) const;

  LogicalResult quantizePerElement(TF::QuantizeV2Op op,
                                   PatternRewriter& rewriter) const;

  LogicalResult quantizePerChannel(TF::QuantizeV2Op op,
                                   PatternRewriter& rewriter) const;
};

// reference implementation:
//   tensorflow/core/kernels/quantize_op.cc
LogicalResult ConvertQuantizeV2Op::adjustMinMaxRange(TF::QuantizeV2Op op,
                                                     PatternRewriter& rewriter,
                                                     Value& minRange,
                                                     Value& maxRange) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.min_range().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));
  Value ensureMinimumRange = rewriter.create<TF::ConstOp>(
      loc,
      DenseFPElementsAttr::get(scalarTensorTy, {op.ensure_minimum_range()}));

  minRange = rewriter.create<TF::MinimumOp>(loc, zeros, op.min_range());
  Value absInputMinRange = rewriter.create<TF::AbsOp>(loc, op.min_range());
  Value absInputMaxRange = rewriter.create<TF::AbsOp>(loc, op.max_range());
  Value maxAbsInputRange =
      rewriter.create<TF::MaximumOp>(loc, absInputMinRange, absInputMaxRange);
  Value epsilon = rewriter.create<TF::MaximumOp>(loc, maxAbsInputRange, ones);
  epsilon = rewriter.create<TF::MulOp>(loc, epsilon, ensureMinimumRange);
  maxRange = rewriter.create<TF::AddOp>(loc, epsilon, minRange);
  maxRange = rewriter.create<TF::MaximumOp>(loc, op.max_range(), maxRange);
  maxRange = rewriter.create<TF::MaximumOp>(loc, zeros, maxRange);

  return success();
}

LogicalResult ConvertQuantizeV2Op::quantizePerElement(
    TF::QuantizeV2Op op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.min_range().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));

  Value minRange, maxRange;
  if (failed(adjustMinMaxRange(op, rewriter, minRange, maxRange)))
    return op->emitError("fail to adjust min/max range\n");

  auto resultTy = op.output().getType().dyn_cast<RankedTensorType>();
  auto resultElemTy = QuantizedTypeToIntegerType(resultTy.getElementType());
  auto limits = GetNumericLimits(resultElemTy);

  if (op.mode() == "MIN_FIRST") {
    Value rangeScale = rewriter.create<TF::SubOp>(loc, maxRange, minRange);
    Value numStepsMinusOne = rewriter.create<TF::ConstOp>(
        loc,
        DenseFPElementsAttr::get(op.min_range().getType(),
                                 {(1ull << resultElemTy.getWidth()) - 1.0f}));
    rangeScale = rewriter.create<TF::DivOp>(loc, rangeScale, numStepsMinusOne);

    Value result = rewriter.create<TF::MulOp>(loc, op.input(), rangeScale);
    result = rewriter.create<TF::RoundOp>(loc, result);

    Value rangeMinScaled =
        rewriter.create<TF::MulOp>(loc, minRange, rangeScale);
    rangeMinScaled = rewriter.create<TF::RoundOp>(loc, rangeMinScaled);

    Value lowestQuantized = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.min_range().getType(),
                                      {limits.lowest_quantized}));
    Value lowerBound = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.min_range().getType(),
                                      {limits.lower_bound_float}));
    Value upperBound = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.min_range().getType(),
                                      {limits.upper_bound_float}));
    Value bias =
        rewriter.create<TF::SubOp>(loc, rangeMinScaled, lowestQuantized);
    result = rewriter.create<TF::SubOp>(loc, result, bias);
    result = rewriter.create<TF::MaximumOp>(loc, result, lowerBound);
    result = rewriter.create<TF::MinimumOp>(loc, result, upperBound);
    result = rewriter.create<TF::CastOp>(loc, op.output().getType(), result);

    SmallVector<Value> newResults{result, minRange, maxRange};
    rewriter.replaceOp(op, newResults);
    return success();
  }

  return op->emitError() << "not supported quantization mode: " << op.mode()
                         << "\n";
}

LogicalResult ConvertQuantizeV2Op::quantizePerChannel(
    TF::QuantizeV2Op op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.min_range().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));

  Value minRange, maxRange;
  if (failed(adjustMinMaxRange(op, rewriter, minRange, maxRange)))
    return op->emitError("fail to adjust min/max range\n");

  auto resultTy = op.output().getType().dyn_cast<RankedTensorType>();
  auto resultElemTy = QuantizedTypeToIntegerType(resultTy.getElementType());
  auto limits = GetNumericLimits(resultElemTy);

  Value minOutputValue = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(
               scalarTensorTy, {limits.lower_bound_float + op.narrow_range()}));
  Value maxOutputValue = rewriter.create<TF::ConstOp>(
      loc,
      DenseFPElementsAttr::get(scalarTensorTy, {limits.upper_bound_float}));
  Value epsilon = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1e-6f}));
  Value maxFloat = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy,
                                    {std::numeric_limits<float>::max()}));
  if (inputMinRangeTy.getRank() != 0) {
    auto rangeShapeType =
        RankedTensorType::get({1}, rewriter.getIntegerType(32));
    Value rangeShape =
        rewriter.create<TF::ShapeOp>(loc, rangeShapeType, minRange);
    maxFloat = rewriter.create<TF::BroadcastToOp>(loc, inputMinRangeTy,
                                                  maxFloat, rangeShape);
  }

  auto boolTensorTy = RankedTensorType::get(inputMinRangeTy.getShape(),
                                            rewriter.getIntegerType(1));

  Value minRangeIsZero =
      rewriter.create<TF::EqualOp>(loc, boolTensorTy, minRange, zeros);
  minRangeIsZero =
      rewriter.create<TF::CastOp>(loc, minRange.getType(), minRangeIsZero);
  Value minEpsilon = rewriter.create<TF::MulOp>(loc, minRangeIsZero, epsilon);
  // minRange is a non-positive number, minus a positive number to make sure the
  // result is not zero.
  Value adjustMinRange = rewriter.create<TF::SubOp>(loc, minRange, minEpsilon);
  Value scaleFactorFromMinSide =
      rewriter.create<TF::DivOp>(loc, minOutputValue, adjustMinRange);
  Value minProduct = rewriter.create<TF::MulOp>(loc, minRange, minOutputValue);
  Value minProductIsPositive =
      rewriter.create<TF::GreaterOp>(loc, boolTensorTy, minProduct, zeros);
  scaleFactorFromMinSide = rewriter.create<TF::SelectOp>(
      loc, scaleFactorFromMinSide.getType(), minProductIsPositive,
      scaleFactorFromMinSide, maxFloat);

  Value maxRangeIsZero =
      rewriter.create<TF::EqualOp>(loc, boolTensorTy, maxRange, zeros);
  maxRangeIsZero =
      rewriter.create<TF::CastOp>(loc, maxRange.getType(), maxRangeIsZero);
  Value maxEpsilon = rewriter.create<TF::MulOp>(loc, maxRangeIsZero, epsilon);
  // maxRange is a non-negative number, minus a positive number to make sure the
  // result is not zero.
  Value adjustMaxRange = rewriter.create<TF::SubOp>(loc, maxRange, maxEpsilon);
  Value scaleFactorFromMaxSide =
      rewriter.create<TF::DivOp>(loc, maxOutputValue, adjustMaxRange);
  Value maxProduct = rewriter.create<TF::MulOp>(loc, maxRange, maxOutputValue);
  Value maxProductIsPositive =
      rewriter.create<TF::GreaterOp>(loc, boolTensorTy, maxProduct, zeros);
  scaleFactorFromMaxSide = rewriter.create<TF::SelectOp>(
      loc, scaleFactorFromMaxSide.getType(), maxProductIsPositive,
      scaleFactorFromMaxSide, maxFloat);

  Value scaleFactor = rewriter.create<TF::MinimumOp>(
      loc, scaleFactorFromMinSide, scaleFactorFromMaxSide);
  minRange = rewriter.create<TF::DivOp>(loc, minOutputValue, scaleFactor);
  maxRange = rewriter.create<TF::DivOp>(loc, maxOutputValue, scaleFactor);

  auto expandDims = [&](Value v) {
    // scalar case, no need to do broadcast.
    if (op.axis() == -1) return v;
    auto scalarIntTensorTy =
        RankedTensorType::get({}, rewriter.getIntegerType(32));
    Value minusOneDim = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(scalarIntTensorTy, {-1}));
    for (int i = op.axis() + 1; i < resultTy.getRank(); ++i) {
      SmallVector<int64_t> newShape;
      auto oldType = v.getType().cast<RankedTensorType>();
      for (int64_t d : oldType.getShape()) newShape.push_back(d);
      newShape.push_back(1);
      auto newType = RankedTensorType::get(newShape, oldType.getElementType());
      v = rewriter.create<TF::ExpandDimsOp>(loc, newType, v, minusOneDim);
    }
    return v;
  };

  Value result = op.input();
  Value expandedMinRange = expandDims(minRange);
  Value expandedMaxRange = expandDims(maxRange);
  Value expandedScaleFactor = expandDims(scaleFactor);
  result = rewriter.create<TF::MaximumOp>(loc, result, expandedMinRange);
  result = rewriter.create<TF::MinimumOp>(loc, result, expandedMaxRange);
  result = rewriter.create<TF::MulOp>(loc, result, expandedScaleFactor);
  if (op.round_mode() == "HALF_AWAY_FROM_ZERO") {
    result = rewriter.create<TF::RoundOp>(loc, result);
  } else {
    return op->emitError() << "not supported round_mode: " << op.round_mode()
                           << "\n";
  }

  result = rewriter.create<TF::CastOp>(loc, op.output().getType(), result);
  SmallVector<Value> newResults{result, minRange, maxRange};
  rewriter.replaceOp(op, newResults);
  return success();
}

#include "tensorflow/compiler/mlir/disc/transforms/lower_tf.inc"

void PrepareTFPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();
  populateWithGenerated(patterns);
  // clang-format off
  patterns.insert<
      ConvertQuantizeV2Op,
      ConvertSqueezeOpDynamic,
      ConvertTopKV2OpDynamic,
      ConvertUniformOp
  >(ctx);
  // clang-format on
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Lower some tf ops before tf2mhlo lowering.
std::unique_ptr<OperationPass<func::FuncOp>> createDiscLowerTfPass() {
  return std::make_unique<PrepareTFPass>();
}

}  // namespace disc_ral
}  // namespace mlir
