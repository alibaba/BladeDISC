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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"                 // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                          // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Matchers.h"                            // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"           // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/disc/IR/disc_tf_additional_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/rng_uniform_custom_call_op.h"
#include "mlir/disc/IR/topk_custom_call_op.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_pdl_utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

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
  PrepareTFPass(const std::string& pdll_files,
                const std::string& pdll_include_dirs)
      : DiscLowerTfPassBase<PrepareTFPass>::DiscLowerTfPassBase() {
    this->pdll_files_ = pdll_files;
    this->pdll_include_dirs_ = pdll_include_dirs;
  }

  // TODO: move to td file
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mhlo_disc::MhloDiscDialect>();
    registry.insert<tensor::TensorDialect>();
    getPDLDependentDialects(registry);
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
                                                   op.getInput());
    } else {
      Location loc = op.getLoc();
      Value input = op.getInput();
      auto input_ty = input.getType().dyn_cast<RankedTensorType>();
      if (!input_ty) {
        return failure();
      }
      int64_t input_rank = input_ty.getRank();
      auto squeeze_dims_attr = op.getSqueezeDims();
      int64_t squeeze_dims_attr_size = squeeze_dims_attr.size();
      if (squeeze_dims_attr_size == 0) {
        return failure();
      }
      llvm::SetVector<int64_t> squeeze_dims;
      for (int64_t i = 0; i < squeeze_dims_attr_size; ++i) {
        int64_t dim = squeeze_dims_attr[i].cast<IntegerAttr>().getInt();
        if (dim < -input_ty.getRank()) return failure();
        if (dim < 0) dim += input_ty.getRank();
        squeeze_dims.insert(dim);
      }
      SmallVector<Value, 4> shape_values;
      int64_t output_rank = input_rank - squeeze_dims_attr_size;
      shape_values.reserve(output_rank);
      for (int64_t i = 0; i < input_rank; ++i) {
        if (squeeze_dims.count(i)) {
          continue;
        }
        auto dim_size = input_ty.getDimSize(i);
        if (dim_size == ShapedType::kDynamic) {
          shape_values.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
        } else {
          shape_values.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, dim_size));
        }
      }
      Value new_shape =
          rewriter.create<tensor::FromElementsOp>(loc, shape_values);
      rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
          op, result_ty, op.getInput(), new_shape);
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
    auto k_type = op.getK().getType().dyn_cast<RankedTensorType>();
    if (!k_type || (k_type.getElementType() != rewriter.getIntegerType(32)) ||
        (k_type.getRank() != 0)) {
      return op.emitOpError() << "TopKV2 requires 0D scalar tensor k";
    }

    auto input_type = op.getInput().getType().dyn_cast<RankedTensorType>();
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
      Value dim =
          rewriter.create<tensor::DimOp>(op.getLoc(), op.getInput(), idx);
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
        ValueRange{op.getInput(), iota_op, op.getK()},
        /*call_target_name*/ "topk",
        /*has_side_effect*/ false,
        /*backend_config*/ StringAttr::get(op->getContext(), ostream.str()));
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
    Value zero = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(op.getDtype(), 0.0));
    Value one = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(op.getDtype(), 1.0));
    auto cfg = PackRandomUniformBackendConfig(
        rewriter.getIntegerAttr(op.getDtype(), op.getSeed()),
        rewriter.getIntegerAttr(op.getDtype(), op.getSeed2()), &rewriter);
    auto custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
        loc, TypeRange{op.getResult().getType()},
        ValueRange{zero, one, op.getShape()},
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

// Expand v in order to be broacast compatiable with the target ranked value.
Value expandValueDims(Value v, int axis, int rank, PatternRewriter& rewriter,
                      Location loc) {
  // Early return since it's already compatible
  if (axis == -1) return v;
  auto scalarIntTensorTy =
      RankedTensorType::get({}, rewriter.getIntegerType(32));
  Value minusOneDim = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarIntTensorTy, {-1}));
  for (int i = axis + 1; i < rank; ++i) {
    auto oldType = v.getType().cast<RankedTensorType>();
    auto newShape = llvm::to_vector(oldType.getShape());
    newShape.push_back(1);
    auto newType = RankedTensorType::get(newShape, oldType.getElementType());
    v = rewriter.create<TF::ExpandDimsOp>(loc, newType, v, minusOneDim);
  }
  return v;
}

struct ConvertQuantizeV2Op : public OpRewritePattern<TF::QuantizeV2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::QuantizeV2Op op,
                                PatternRewriter& rewriter) const final {
    auto inputTy = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputMinRangeTy =
        op.getMinRange().getType().dyn_cast<RankedTensorType>();
    auto inputMaxRangeTy =
        op.getMaxRange().getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.getOutput().getType().dyn_cast<RankedTensorType>();

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
    if (op.getAxis() != -1 && op.getMode() == "MIN_FIRST") return failure();

    if (op.getAxis() == -1 && op.getMode() == "MIN_FIRST") {
      return quantizeModeMinFirst(op, rewriter);
    }

    // TODO(disc): support `MIN_COMBINED` mode
    if (op.getMode() == "MIN_COMBINED") return failure();

    // TODO(disc): support `HALF_TO_EVEN` mode
    if (op.getRoundMode() == "HALF_TO_EVEN") return failure();

    return quantizeModeScaledOrMinCombined(op, rewriter);
  }

  LogicalResult adjustMinMaxRange(TF::QuantizeV2Op op,
                                  PatternRewriter& rewriter, Value& minRange,
                                  Value& maxRange) const;

  LogicalResult quantizeModeMinFirst(TF::QuantizeV2Op op,
                                     PatternRewriter& rewriter) const;

  LogicalResult quantizeModeScaledOrMinCombined(
      TF::QuantizeV2Op op, PatternRewriter& rewriter) const;
};

// reference implementation:
//   tensorflow/core/kernels/quantize_op.cc
LogicalResult ConvertQuantizeV2Op::adjustMinMaxRange(TF::QuantizeV2Op op,
                                                     PatternRewriter& rewriter,
                                                     Value& minRange,
                                                     Value& maxRange) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.getMinRange().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));
  Value ensureMinimumRange = rewriter.create<TF::ConstOp>(
      loc,
      DenseFPElementsAttr::get(scalarTensorTy, {op.getEnsureMinimumRange()}));

  minRange = rewriter.create<TF::MinimumOp>(loc, zeros, op.getMinRange());
  Value absInputMinRange = rewriter.create<TF::AbsOp>(loc, op.getMinRange());
  Value absInputMaxRange = rewriter.create<TF::AbsOp>(loc, op.getMaxRange());
  Value maxAbsInputRange =
      rewriter.create<TF::MaximumOp>(loc, absInputMinRange, absInputMaxRange);
  Value epsilon = rewriter.create<TF::MaximumOp>(loc, maxAbsInputRange, ones);
  epsilon = rewriter.create<TF::MulOp>(loc, epsilon, ensureMinimumRange);
  maxRange = rewriter.create<TF::AddOp>(loc, epsilon, minRange);
  maxRange = rewriter.create<TF::MaximumOp>(loc, op.getMaxRange(), maxRange);
  maxRange = rewriter.create<TF::MaximumOp>(loc, zeros, maxRange);

  return success();
}

LogicalResult ConvertQuantizeV2Op::quantizeModeMinFirst(
    TF::QuantizeV2Op op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.getMinRange().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));

  Value minRange, maxRange;
  if (failed(adjustMinMaxRange(op, rewriter, minRange, maxRange)))
    return op->emitError("fail to adjust min/max range\n");

  auto resultTy = op.getOutput().getType().dyn_cast<RankedTensorType>();
  auto resultElemTy = QuantizedTypeToIntegerType(resultTy.getElementType());
  auto limits = GetNumericLimits(resultElemTy);

  if (op.getMode() == "MIN_FIRST") {
    Value rangeScale = rewriter.create<TF::SubOp>(loc, maxRange, minRange);
    Value numStepsMinusOne = rewriter.create<TF::ConstOp>(
        loc,
        DenseFPElementsAttr::get(op.getMinRange().getType(),
                                 {(1ull << resultElemTy.getWidth()) - 1.0f}));
    rangeScale = rewriter.create<TF::DivOp>(loc, rangeScale, numStepsMinusOne);

    Value result = rewriter.create<TF::MulOp>(loc, op.getInput(), rangeScale);
    result = rewriter.create<TF::RoundOp>(loc, result);

    Value rangeMinScaled =
        rewriter.create<TF::MulOp>(loc, minRange, rangeScale);
    rangeMinScaled = rewriter.create<TF::RoundOp>(loc, rangeMinScaled);

    Value lowestQuantized = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.getMinRange().getType(),
                                      {limits.lowest_quantized}));
    Value lowerBound = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.getMinRange().getType(),
                                      {limits.lower_bound_float}));
    Value upperBound = rewriter.create<TF::ConstOp>(
        loc, DenseFPElementsAttr::get(op.getMinRange().getType(),
                                      {limits.upper_bound_float}));
    Value bias =
        rewriter.create<TF::SubOp>(loc, rangeMinScaled, lowestQuantized);
    result = rewriter.create<TF::SubOp>(loc, result, bias);
    result = rewriter.create<TF::MaximumOp>(loc, result, lowerBound);
    result = rewriter.create<TF::MinimumOp>(loc, result, upperBound);
    result = rewriter.create<TF::CastOp>(loc, op.getOutput().getType(), result);

    SmallVector<Value> newResults{result, minRange, maxRange};
    rewriter.replaceOp(op, newResults);
    return success();
  }

  return op->emitError() << "not supported quantization mode: " << op.getMode()
                         << "\n";
}

LogicalResult ConvertQuantizeV2Op::quantizeModeScaledOrMinCombined(
    TF::QuantizeV2Op op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.getMinRange().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  Value zeros = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {0.0f}));
  Value ones = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {1.0f}));

  Value minRange, maxRange;
  if (failed(adjustMinMaxRange(op, rewriter, minRange, maxRange)))
    return op->emitError("fail to adjust min/max range\n");

  auto resultTy = op.getOutput().getType().dyn_cast<RankedTensorType>();
  auto resultElemTy = QuantizedTypeToIntegerType(resultTy.getElementType());
  auto limits = GetNumericLimits(resultElemTy);

  Value minOutputValue = rewriter.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarTensorTy, {limits.lower_bound_float +
                                                     op.getNarrowRange()}));
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
    return expandValueDims(v, op.getAxis(), resultTy.getRank(), rewriter, loc);
  };

  Value result = op.getInput();
  Value expandedMinRange = expandDims(minRange);
  Value expandedMaxRange = expandDims(maxRange);
  Value expandedScaleFactor = expandDims(scaleFactor);
  result = rewriter.create<TF::MaximumOp>(loc, result, expandedMinRange);
  result = rewriter.create<TF::MinimumOp>(loc, result, expandedMaxRange);
  result = rewriter.create<TF::MulOp>(loc, result, expandedScaleFactor);
  if (op.getRoundMode() == "HALF_AWAY_FROM_ZERO") {
    result = rewriter.create<TF::RoundOp>(loc, result);
  } else {
    return op->emitError() << "not supported round_mode: " << op.getRoundMode()
                           << "\n";
  }

  result = rewriter.create<TF::CastOp>(loc, op.getOutput().getType(), result);
  SmallVector<Value> newResults{result, minRange, maxRange};
  rewriter.replaceOp(op, newResults);
  return success();
}

struct ConvertDequantizeOp : public OpRewritePattern<TF::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::DequantizeOp op,
                                PatternRewriter& rewriter) const final {
    auto inputTy = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputMinRangeTy =
        op.getMinRange().getType().dyn_cast<RankedTensorType>();
    auto inputMaxRangeTy =
        op.getMaxRange().getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.getOutput().getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !inputMinRangeTy || !inputMaxRangeTy || !resultTy)
      return failure();

    auto inputElemTy = QuantizedTypeToIntegerType(inputTy.getElementType());
    if (!inputElemTy) return failure();

    // TODO(disc): support other float type
    if (!resultTy.getElementType().isF32() ||
        !inputMinRangeTy.getElementType().isF32() ||
        !inputMaxRangeTy.getElementType().isF32()) {
      return failure();
    }

    // TODO(disc): support `MIN_FIRST` mode
    if (op.getMode() == "MIN_FIRST") return failure();

    // TODO(disc): support `MIN_COMBINED` mode
    if (op.getMode() == "MIN_COMBINED") return failure();

    return dequantize(op, rewriter);
  }

  LogicalResult dequantize(TF::DequantizeOp op,
                           PatternRewriter& rewriter) const;
};

LogicalResult ConvertDequantizeOp::dequantize(TF::DequantizeOp op,
                                              PatternRewriter& rewriter) const {
  Location loc = op.getLoc();

  auto inputMinRangeTy = op.getMinRange().getType().cast<RankedTensorType>();
  auto scalarTensorTy =
      RankedTensorType::get({}, inputMinRangeTy.getElementType());

  auto inputTy = op.getInput().getType().dyn_cast<RankedTensorType>();
  auto inputElemTy = QuantizedTypeToIntegerType(inputTy.getElementType());
  auto resultTy = op.getOutput().getType().dyn_cast<RankedTensorType>();
  auto limits = GetNumericLimits(inputElemTy);

  Value minRange = op.getMinRange();
  Value maxRange = op.getMaxRange();
  Value input = op.getInput();
  auto newInputTy =
      RankedTensorType::get(inputTy.getShape(), resultTy.getElementType());
  input = rewriter.create<TF::CastOp>(loc, newInputTy, input);

  Value maxOutputValue = rewriter.create<TF::ConstOp>(
      loc,
      DenseFPElementsAttr::get(scalarTensorTy, {limits.upper_bound_float}));

  if (op.getMode() == "SCALED") {
    // `tf.Dequantive` and `_MklDequantize` have different behaviours.
    // We have to disable onednn before testing `tf.Dequantive` since
    // `tf.Dequantive` will be converted to `_MklDequantize` by default when
    // onednn is enabled.
    // TODO(disc): figure out how to combine the TF version and the MKL version.
    Value minOutputValue = rewriter.create<TF::ConstOp>(
        loc,
        DenseFPElementsAttr::get(
            scalarTensorTy, {limits.lower_bound_float + op.getNarrowRange()}));
    Value scaleFactorFromMax =
        rewriter.create<TF::DivOp>(loc, maxRange, maxOutputValue);
    Value scaleFactor = scaleFactorFromMax;
    if (limits.lower_bound_float != 0.f) {
      Value scaleFactorFromMin =
          rewriter.create<TF::DivOp>(loc, minRange, minOutputValue);
      scaleFactor = rewriter.create<TF::MaximumOp>(loc, scaleFactorFromMin,
                                                   scaleFactorFromMax);
    }
    scaleFactor = expandValueDims(scaleFactor, op.getAxis(), inputTy.getRank(),
                                  rewriter, loc);
    Value result = rewriter.create<TF::MulOp>(loc, scaleFactor, input);
    result.setType(op.getOutput().getType());
    rewriter.replaceOp(op, {result});
    return success();
  }

  return op->emitError() << "not supported quantization mode: " << op.getMode()
                         << "\n";
}

// TODO(wyzero): Deprecated. It'll be removed after we change the behavior
// of the fake_quant op.
struct ConvertFakeQuantOp : public OpRewritePattern<TF::DiscFakeQuantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::DiscFakeQuantOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getUseDynamic()) {
      return op->emitError("dynamic quantization is not supported yet");
    }
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto scale = op.getScale();
    auto zero_point = op.getZeroPoint();
    SmallVector<int64_t> axis;
    for (auto& v : op.getAxis()) {
      axis.push_back(v.cast<IntegerAttr>().getInt());
    }
    // Default mode of round in tf is RoundHalfAwayFromZero
    // todo(disc): should read it from TF::DiscFakeQuantOp
    auto round_mode_attr = mlir::mhlo_disc::RoundModeEnumAttr::get(
        rewriter.getContext(),
        mlir::mhlo_disc::RoundModeEnum::RoundHalfAwayFromZero);
    Value new_op = rewriter.create<mhlo_disc::FakeQuantOp>(
        loc, op.getType(), input, scale, zero_point, op.getUseSignedAttr(),
        op.getUseSymmetricAttr(), GetI64ElementsAttr(axis, &rewriter),
        op.getNumBitsAttr(), op.getQuantMinAttr(), op.getQuantMaxAttr(),
        op.getUseDynamicAttr(), round_mode_attr);
    rewriter.replaceOp(op, {new_op});
    return success();
  }
};

// Directly lower TF::FakeQuantOp to mhlo_disc::Quant + mhlo_disc::Dequant.
struct ConvertFakeQuantV2Op : public OpRewritePattern<TF::DiscFakeQuantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::DiscFakeQuantOp op,
                                PatternRewriter& rewriter) const final {
    Location loc = op.getLoc();
    auto inputTy = op.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputTy)
      return op->emitError("Only support input with ranked tensor type");

    SmallVector<int64_t> axis;
    for (auto& v : op.getAxis()) {
      axis.push_back(v.cast<IntegerAttr>().getInt());
    }

    int64_t numBits = op.getNumBitsAttr().getInt();
    auto quantizedElemTy =
        op.getUseSignedAttr().getValue()
            ? IntegerType::get(rewriter.getContext(), numBits)
            : IntegerType::get(
                  rewriter.getContext(), numBits,
                  mlir::IntegerType::SignednessSemantics::Unsigned);
    auto quantizedTy = RankedTensorType::get(
        inputTy.getShape(), quantizedElemTy, inputTy.getEncoding());

    // Default mode of round in tf is RoundHalfAwayFromZero
    // todo(disc): should read it from TF::DiscFakeQuantOp
    auto round_mode_attr = mlir::mhlo_disc::RoundModeEnumAttr::get(
        rewriter.getContext(),
        mlir::mhlo_disc::RoundModeEnum::RoundHalfAwayFromZero);

    Value quantizedValue = rewriter.create<mhlo_disc::QuantizeOp>(
        loc, quantizedTy, op.getInput(), op.getScale(), op.getZeroPoint(),
        op.getUseSymmetricAttr(), GetI64ElementsAttr(axis, &rewriter),
        op.getQuantMinAttr(), op.getQuantMaxAttr(), op.getUseDynamicAttr(),
        round_mode_attr);
    Value dequantizedValue = rewriter.create<mhlo_disc::DequantizeOp>(
        loc, op.getType(), quantizedValue, op.getScale(), op.getZeroPoint(),
        op.getUseSymmetricAttr(), GetI64ElementsAttr(axis, &rewriter),
        op.getUseDynamicAttr(), round_mode_attr);
    rewriter.replaceOp(op, {dequantizedValue});
    return success();
  }
};

Type ToLegalElementType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<mlir::TF::Qint8Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 8);
      })
      .Case<mlir::TF::Qint16Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 16);
      })
      .Case<mlir::TF::Qint32Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 32);
      })
      .Case<mlir::TF::Quint8Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 8,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<mlir::TF::Quint16Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 16,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

bool isQint32ConstantZeroTensor(Value v) {
  auto constOp = dyn_cast_or_null<TF::ConstOp>(v.getDefiningOp());
  if (!constOp) return false;

  tensorflow::Tensor out;
  if (tensorflow::ConvertToTensor(constOp.getValue(), &out) !=
      tensorflow::OkStatus())
    return false;

  int64_t numElems = out.NumElements();
  auto flat = out.bit_casted_shaped<int32_t, 1>({numElems});
  for (int64_t i = 0; i < numElems; ++i) {
    if (flat(i) != 0) return false;
  }
  // in case the value is an empty tensor.
  return (numElems != 0);
}

bool isConstantZeroTensor(Value v) {
  auto castOp = dyn_cast_or_null<TF::CastOp>(v.getDefiningOp());
  if (castOp) return isConstantZeroTensor(castOp->getOperand(0));

  auto ty = v.getType().dyn_cast<RankedTensorType>();
  if (ty && ty.getElementType().isa<mlir::TF::Qint32Type>()) {
    return isQint32ConstantZeroTensor(v);
  }

  DenseElementsAttr denseAttr;
  if (!matchPattern(v, m_Constant(&denseAttr))) return false;
  if (denseAttr.getNumElements() != 1 && !denseAttr.isSplat()) return false;

  Type elemTy = denseAttr.getElementType();
  if (elemTy.isIntOrIndex()) {
    return (*denseAttr.getValues<APInt>().begin()).getSExtValue() == 0;
  } else if (elemTy.isa<FloatType>()) {
    return (*denseAttr.getValues<APFloat>().begin()).convertToDouble() == 0;
  }
  return false;
}

struct ConvParams {
  Value input;
  Value filter;
  Value output;
  Value padding;
  Operation* op;
  Attribute config;
};

LogicalResult getPaddingValues(Operation* op, OpBuilder& rewriter,
                               Value input_size, Value filter_size,
                               int64_t dilation_rate, int64_t stride,
                               tensorflow::Padding padding_type,
                               Type shape_scalar_type, Value* padding_low,
                               Value* padding_high) {
  // Stride must be > 0
  if (stride <= 0)
    return op->emitError() << "stride of conv is suppose positive but got: "
                           << stride << "\n";
  // Dilation rate must be >= 1
  if (dilation_rate < 1)
    return op->emitError()
           << "dilation_rate of conv must be large than zero but got: "
           << dilation_rate << "\n";

  Location loc = op->getLoc();
  switch (padding_type) {
    case tensorflow::Padding::VALID: {
      auto zero =
          rewriter.create<arith::ConstantIntOp>(loc, 0, shape_scalar_type);
      *padding_low = *padding_high = zero;
      break;
    }
    case tensorflow::Padding::EXPLICIT:
      break;
    case tensorflow::Padding::SAME: {
      auto zero =
          rewriter.create<arith::ConstantIntOp>(loc, 0, shape_scalar_type);
      auto one =
          rewriter.create<arith::ConstantIntOp>(loc, 1, shape_scalar_type);
      auto two =
          rewriter.create<arith::ConstantIntOp>(loc, 2, shape_scalar_type);
      // See also the parallel implementation in
      // GetWindowedOutputSizeFromDimsV2. effective_filter_size = (filter_size
      // - 1) * dilation_rate + 1
      Value stride_value =
          rewriter.create<arith::ConstantIntOp>(loc, stride, shape_scalar_type);
      Value dilation_rate_value = rewriter.create<arith::ConstantIntOp>(
          loc, dilation_rate, shape_scalar_type);
      Value effective_filter_size_op = rewriter.create<arith::AddIOp>(
          loc, one,
          rewriter.create<arith::MulIOp>(
              loc, dilation_rate_value,
              rewriter.create<arith::SubIOp>(loc, filter_size, one)));
      // output_size = (input_size + stride - 1) / stride;
      Value output_size = rewriter.create<arith::DivUIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, input_size,
              rewriter.create<arith::SubIOp>(loc, stride_value, one)),
          stride_value);
      // std::max(int64{0}, (output_size - 1) * stride +
      //     effective_filter_size - input_size);
      Value padding_needed = rewriter.create<arith::SubIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, effective_filter_size_op,
              rewriter.create<arith::MulIOp>(
                  loc, stride_value,
                  rewriter.create<arith::SubIOp>(loc, output_size, one))),
          input_size);
      Value cond = rewriter.create<mlir::arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, padding_needed, zero);
      padding_needed = rewriter.create<mlir::arith::SelectOp>(
          loc, padding_needed.getType(), cond, padding_needed, zero);
      *padding_low = rewriter.create<arith::DivUIOp>(loc, padding_needed, two);
      *padding_high =
          rewriter.create<arith::SubIOp>(loc, padding_needed, *padding_low);
      break;
    }
  }
  return success();
}

// Returns size of dimension at the specified index, if ranked tensor.
// Otherwise, returns -1.
//
// Aborts if the type is ranked but doesn't have the dimension.
int64_t getDimSize(Type ty, int64_t index) {
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return -1;

  return ranked_ty.getDimSize(index);
}

NamedAttribute getConvDimensionNumbersAttr(
    ArrayRef<int64_t> spatial_dims, tensorflow::TensorFormat format,
    Builder* builder,
    tensorflow::FilterTensorFormat filter_format = tensorflow::FORMAT_HWIO) {
  int64_t num_spatial_dims = spatial_dims.size();
  int64_t num_dims = num_spatial_dims + 2;

  int64_t batch_dim = GetTensorBatchDimIndex(num_dims, format);
  int64_t feature_dim = GetTensorFeatureDimIndex(num_dims, format);

  int64_t kernel_input_feature_dim = num_spatial_dims;
  int64_t kernel_output_feature_dim = num_spatial_dims + 1;
  SmallVector<int64_t, 4> kernel_spatial_dimensions;
  kernel_spatial_dimensions.resize(num_spatial_dims);
  if (filter_format == tensorflow::FORMAT_OHWI) {
    kernel_input_feature_dim = num_spatial_dims + 1;
    kernel_output_feature_dim = 0;
    std::iota(kernel_spatial_dimensions.begin(),
              kernel_spatial_dimensions.end(), 1);
  } else {
    assert(filter_format == tensorflow::FORMAT_HWIO);
    // Filters data_format is always HWIO so input channels dimension is after
    // all spatial dimensions.
    std::iota(kernel_spatial_dimensions.begin(),
              kernel_spatial_dimensions.end(), 0);
  }

  return builder->getNamedAttr(
      "dimension_numbers",
      mhlo::ConvDimensionNumbersAttr::get(
          builder->getContext(), batch_dim, feature_dim, spatial_dims,
          kernel_input_feature_dim, kernel_output_feature_dim,
          kernel_spatial_dimensions, batch_dim, feature_dim, spatial_dims));
}

LogicalResult parseConvParams(ConvParams& params, OpBuilder& b, Location& loc) {
  Operation* op = params.op;
  auto input_ty = params.input.getType().dyn_cast<RankedTensorType>();
  auto filter_ty = params.filter.getType().dyn_cast<RankedTensorType>();
  int num_dims = input_ty.getRank();
  int64_t num_spatial_dims = num_dims - 2;
  Type shape_scalar_type = b.getIntegerType(32);

  tensorflow::TensorFormat data_format = tensorflow::TensorFormat::FORMAT_NHWC;
  tensorflow::Padding padding;
  auto paddingStr = op->getAttrOfType<StringAttr>("padding").str();
  if (!tensorflow::GetPaddingFromString(paddingStr, &padding).ok())
    return op->emitError() << "fail to parse padding attributes\n";

  ArrayRef<Attribute> dilations =
      op->getAttrOfType<ArrayAttr>("dilations").getValue();
  ArrayRef<Attribute> strides =
      op->getAttrOfType<ArrayAttr>("strides").getValue();
  ArrayRef<Attribute> explicit_paddings;
  if (padding == tensorflow::Padding::EXPLICIT) {
    // EXPLICIT padding mode and the associated attribute is attached to
    // Conv2D.
    explicit_paddings = op->getAttrOfType<ArrayAttr>("padding_list").getValue();
  }

  SmallVector<int64_t> spatial_dim_indices;
  SmallVector<int64_t> rhs_dilations;
  SmallVector<int64_t> window_strides;
  SmallVector<Value> paddings;
  auto get_int = [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  };
  auto get_const = [&](int64_t val) {
    return b.create<mlir::arith::ConstantIntOp>(loc, val, shape_scalar_type);
  };
  auto get_dim_value = [&](Value val, int64_t dim) {
    Value dim_value = b.create<tensor::DimOp>(loc, val, dim);
    return b.create<arith::IndexCastOp>(loc, shape_scalar_type, dim_value);
  };

  for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
    const int64_t dim =
        tensorflow::GetTensorSpatialDimIndex(num_dims, data_format, i);
    spatial_dim_indices.push_back(dim);

    const int64_t dilation = get_int(dilations[dim]);
    rhs_dilations.push_back(dilation);
    const int64_t stride = get_int(strides[dim]);
    window_strides.push_back(stride);

    Value pad_low, pad_high;
    if (padding == tensorflow::Padding::EXPLICIT) {
      pad_low = get_const(get_int(explicit_paddings[2 * dim]));
      pad_high = get_const(get_int(explicit_paddings[2 * dim + 1]));
    } else {
      auto input_size = get_dim_value(params.input, dim);
      // filter layout OHWI
      auto filter_size = get_dim_value(params.filter, i + 1);
      if (failed(getPaddingValues(op, b, input_size, filter_size, dilation,
                                  stride, padding, shape_scalar_type, &pad_low,
                                  &pad_high))) {
        return failure();
      }
    }
    paddings.push_back(pad_low);
    paddings.push_back(pad_high);
  }

  auto rhs_dilations_attr =
      b.getNamedAttr("rhs_dilation", GetI64ElementsAttr(rhs_dilations, &b));

  auto window_strides_attr =
      b.getNamedAttr("window_strides", GetI64ElementsAttr(window_strides, &b));

  auto dimension_numbers_attr = getConvDimensionNumbersAttr(
      spatial_dim_indices, data_format, &b, tensorflow::FORMAT_OHWI);

  const int64_t input_channels = getDimSize(
      input_ty, tensorflow::GetTensorFeatureDimIndex(num_dims, data_format));
  // Filters data_format is always HWIO so input channels dimension is after
  // all spatial dimensions.
  const int64_t filter_channels = getDimSize(filter_ty, num_spatial_dims);
  // TensorFlow convolution op verifies that the number of input channels is
  // divisible by the number of filter channels.
  // For depthwise convolution the feature_group_count argument would be set
  // to the input feature dimension.
  const int64_t feature_group_count = input_channels / filter_channels;
  auto feature_group_count_attr = b.getNamedAttr(
      "feature_group_count", b.getI64IntegerAttr(feature_group_count));

  auto batch_group_count_attr =
      b.getNamedAttr("batch_group_count", b.getI64IntegerAttr(1));

  params.padding = b.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get(2 * num_spatial_dims, b.getI32Type()),
      paddings);

  NamedAttribute attrs[] = {rhs_dilations_attr, window_strides_attr,
                            dimension_numbers_attr, feature_group_count_attr,
                            batch_group_count_attr};
  params.config = b.getDictionaryAttr(attrs);

  return success();
}

LogicalResult matchAndRwriterQuantizedConv2DWithBiasAndRequantizeOp(
    Operation* op) {
  if (op->getNumOperands() != 9 || op->getNumResults() != 3)
    return op->emitError() << "mismatch # operands or # results\n";

  Value input = op->getOperand(0);
  Value filter = op->getOperand(1);
  Value bias = op->getOperand(2);
  Value minInput = op->getOperand(3);
  Value maxInput = op->getOperand(4);
  Value minFilter = op->getOperand(5);
  Value maxFilter = op->getOperand(6);
  Value minFreezedOutput = op->getOperand(7);
  Value maxFreezedOutput = op->getOperand(8);
  Value output = op->getResult(0);
  Value minOutput = op->getResult(1);
  Value maxOutput = op->getResult(2);

  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto filterTy = filter.getType().dyn_cast<RankedTensorType>();
  auto biasTy = bias.getType().dyn_cast<RankedTensorType>();
  auto minInputTy = minInput.getType().dyn_cast<RankedTensorType>();
  auto maxInputTy = maxInput.getType().dyn_cast<RankedTensorType>();
  auto minFilterTy = minFilter.getType().dyn_cast<RankedTensorType>();
  auto maxFilterTy = maxFilter.getType().dyn_cast<RankedTensorType>();
  auto minFreezedOutputTy =
      minFreezedOutput.getType().dyn_cast<RankedTensorType>();
  auto maxFreezedOutputTy =
      maxFreezedOutput.getType().dyn_cast<RankedTensorType>();
  auto outputTy = output.getType().dyn_cast<RankedTensorType>();
  auto minOutputTy = minOutput.getType().dyn_cast<RankedTensorType>();
  auto maxOutputTy = maxOutput.getType().dyn_cast<RankedTensorType>();

  // only try to match the op has static rank.
  if (!inputTy || !filterTy || !biasTy || !minInputTy || !maxInputTy ||
      !minFilterTy || !maxFilterTy || !minFreezedOutputTy ||
      !maxFreezedOutputTy || !outputTy || !minOutputTy || !maxOutputTy)
    return success();

  // Currently only support const zero bias (a.k.a no bias at all)
  // TODO(disc): support non-zero bias.
  if (!isConstantZeroTensor(bias)) return success();

  // Currently only support s8s8s8 config.
  // TODO(disc): support other configs.
  if (!inputTy.getElementType().isa<TF::Qint8Type>() ||
      !filterTy.getElementType().isa<TF::Qint8Type>() ||
      !outputTy.getElementType().isa<TF::Qint8Type>())
    return success();

  OpBuilder b(op);
  Location loc = op->getLoc();

  ConvParams params;
  params.op = op;
  auto newInputTy =
      RankedTensorType::get(inputTy.getShape(), b.getIntegerType(8));
  params.input = b.create<TF::CastOp>(loc, newInputTy, input);
  auto newFilterTy =
      RankedTensorType::get(filterTy.getShape(), b.getIntegerType(8));
  params.filter = b.create<TF::CastOp>(loc, newFilterTy, filter);
  // filter layout conversion: HWIO -> OHWI
  int rank = newFilterTy.getRank();
  SmallVector<int> permutation(rank);
  SmallVector<int64_t> newFilterShape(rank);
  permutation[0] = rank - 1;
  newFilterShape[0] = filterTy.getShape()[rank - 1];
  permutation[rank - 1] = rank - 2;
  newFilterShape[rank - 1] = filterTy.getShape()[rank - 2];
  for (int i = 1; i < rank - 1; ++i) {
    permutation[i] = i - 1;
    newFilterShape[i] = filterTy.getShape()[i - 1];
  }
  auto permTensorTy =
      RankedTensorType::get({permutation.size()}, b.getIntegerType(32));
  Value permutationValue = b.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(permTensorTy, permutation));
  newFilterTy =
      RankedTensorType::get(newFilterShape, newFilterTy.getElementType());
  params.filter = b.create<TF::TransposeOp>(loc, newFilterTy, params.filter,
                                            permutationValue);

  if (failed(parseConvParams(params, b, loc))) return failure();

  Value absMinInput = b.create<TF::AbsOp>(loc, minInput);
  Value absMaxInput = b.create<TF::AbsOp>(loc, maxInput);
  Value inputRange = b.create<TF::MaximumOp>(loc, absMinInput, absMaxInput);
  Value absMinOutput = b.create<TF::AbsOp>(loc, minFreezedOutput);
  Value absMaxOutput = b.create<TF::AbsOp>(loc, maxFreezedOutput);
  Value outputRange = b.create<TF::MaximumOp>(loc, absMinOutput, absMaxOutput);
  Value absMinFilter = b.create<TF::AbsOp>(loc, minFilter);
  Value absMaxFilter = b.create<TF::AbsOp>(loc, maxFilter);
  Value filterRange = b.create<TF::MaximumOp>(loc, absMinFilter, absMaxFilter);

  auto scalarFpTensorTy =
      RankedTensorType::get({}, minInputTy.getElementType());
  Value s8Limit = b.create<TF::ConstOp>(
      loc, DenseFPElementsAttr::get(scalarFpTensorTy, {127.0f}));

  Value inputScales = b.create<TF::DivOp>(loc, inputRange, s8Limit);
  Value filterScales = b.create<TF::DivOp>(loc, filterRange, s8Limit);
  Value outputScales = b.create<TF::DivOp>(loc, outputRange, s8Limit);

  SmallVector<Value> newOperands{params.input, params.filter, params.padding,
                                 inputScales,  filterScales,  outputScales};

  auto newOutputTy =
      RankedTensorType::get(outputTy.getShape(), b.getIntegerType(8));
  auto customOp = b.create<mhlo_disc::CustomCallOp>(
      loc, newOutputTy, newOperands,
      /*call_target_name*/ "ral_qconv_s8_s8_s8",
      /*has_side_effect*/ false,
      /*backend_config*/ params.config);

  auto newOutput = b.create<TF::CastOp>(loc, outputTy, customOp->getResult(0));
  op->getResult(0).replaceAllUsesWith(newOutput);
  op->getResult(1).replaceAllUsesWith(minFreezedOutput);
  op->getResult(2).replaceAllUsesWith(maxFreezedOutput);
  op->erase();
  return success();
}

LogicalResult convertQuantizedConv2DWithBiasAndRequantizeOp(func::FuncOp func) {
  SmallVector<Operation*> qconvOps;
  func.walk([&](Operation* op) {
    if (op->getName().stripDialect().str() ==
        "QuantizedConv2DWithBiasAndRequantize")
      qconvOps.push_back(op);
  });

  for (Operation* op : qconvOps) {
    if (failed(matchAndRwriterQuantizedConv2DWithBiasAndRequantizeOp(op)))
      return failure();
  }
  return success();
}

// op {
//   name: "Bucketize"
//   input_arg {
//     name: "input"
//     type_attr: "T"
//   }
//   output_arg {
//     name: "output"
//     type: DT_INT32
//   }
//   attr {
//     name: "T"
//     type: "type"
//     allowed_values {
//       list {
//         type: DT_INT32
//         type: DT_INT64
//         type: DT_FLOAT
//         type: DT_DOUBLE
//       }
//     }
//   }
//   attr {
//     name: "boundaries"
//     type: "list(float)"
//   }
// }
class ConvertBucketizeOp : public OpRewritePattern<TF::BucketizeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::BucketizeOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    // attr: boundaries, type is float according op definition
    auto boundaries =
        rewriter
            .create<mhlo::ConstantOp>(
                loc, GetF32ElementsAttr(op.getBoundaries(), &rewriter))
            .getResult();
    // the following behavior matches the behavior of the core
    // Bucketize kernel. However, comparing an int32 or int64 against float may
    // lead to inaccurate bucketing due to rounding.
    if (input_type.isF64()) {
      boundaries = rewriter.create<mhlo::ConvertOp>(loc, boundaries,
                                                    rewriter.getF64Type());
    } else {
      input =
          rewriter.create<mhlo::ConvertOp>(loc, input, rewriter.getF32Type());
      input_type = input.getType().dyn_cast<RankedTensorType>();
    }

    int64_t input_rank = input_type.getRank();
    SmallVector<Value, 4> broadcast_to_shape;
    broadcast_to_shape.reserve(input_rank + 1);
    SmallVector<int64_t, 4> broadcast_shape(input_rank + 1,
                                            ShapedType::kDynamic);
    for (int i = 0; i < input_rank; ++i) {
      int64_t dim_size = input_type.getDimSize(i);
      if (dim_size != ShapedType::kDynamic) {
        broadcast_shape[i] = dim_size;
      }
      broadcast_to_shape.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(),
          rewriter.create<tensor::DimOp>(loc, input, i)));
    }
    auto array_attr = op.getBoundaries().cast<ArrayAttr>();
    broadcast_shape[input_rank] = array_attr.getValue().size();
    broadcast_to_shape.push_back(rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI32Type(),
                                     array_attr.getValue().size())));

    RankedTensorType broadcast_type =
        RankedTensorType::get(broadcast_shape, input_type.getElementType());

    Value broadcast_to_shape_tensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(broadcast_to_shape.size())},
                              rewriter.getI32Type()),
        broadcast_to_shape);

    auto broadcast_dims = GetI64ElementsAttrForSeq(0, input_rank, &rewriter);

    // like [4] --> [2, 10, 4]
    SmallVector<int64_t> boradcast_dim;
    boradcast_dim.push_back(static_cast<int64_t>(input_rank));
    boundaries = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, broadcast_type, boundaries, broadcast_to_shape_tensor,
        GetI64ElementsAttr(boradcast_dim, &rewriter));

    input = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, broadcast_type, input, broadcast_to_shape_tensor, broadcast_dims);
    auto comparison = rewriter.create<mhlo::CompareOp>(
        loc, input, boundaries, mhlo::ComparisonDirection::GE);
    // output should be int32 type
    auto convert = rewriter.create<mhlo::ConvertOp>(
        loc, comparison, rewriter.getIntegerType(32));

    // reduce
    Type element_type = getElementTypeOrSelf(convert.getType());
    Value zero = mhlo::GetScalarConstOfType(element_type, loc, 0, &rewriter);
    SmallVector<int64_t> reduce_dim;
    reduce_dim.push_back(static_cast<int64_t>(input_rank));
    auto buckets = rewriter.create<mhlo::ReduceOp>(
        loc, convert.getResult(), zero,
        GetI64ElementsAttr(reduce_dim, &rewriter));
    mhlo::BuildReduceBody<mhlo::AddOp>(element_type, &buckets.getBody(),
                                       &rewriter);

    rewriter.replaceOp(op, {buckets.getResult(0)});
    return success();
  }
};

class ConvertSparseReshapeOp : public OpRewritePattern<TF::SparseReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SparseReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto hlo_sparse_reshape = rewriter.create<mhlo_disc::SparseReshapeOp>(
        loc, op.getOutputIndices().getType(), op.getOutputShape().getType(),
        op.getInputIndices(), op.getInputShape(), op.getNewShape());
    rewriter.replaceOp(op, hlo_sparse_reshape.getResults());
    return success();
  }
};

class ConvertSparseFillEmptyRowsOp
    : public OpRewritePattern<TF::SparseFillEmptyRowsOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SparseFillEmptyRowsOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto indices_type = op.getIndices().getType().dyn_cast<RankedTensorType>();
    auto values_type = op.getValues().getType().dyn_cast<RankedTensorType>();
    auto dense_shape_type =
        op.getDenseShape().getType().dyn_cast<RankedTensorType>();
    auto default_value_type =
        op.getDefaultValue().getType().dyn_cast<RankedTensorType>();
    if (!indices_type || !values_type || !dense_shape_type ||
        !default_value_type) {
      return failure();
    }
    auto element_type = values_type.getElementType();
    auto sparse_fill_empty_rows_op =
        rewriter.create<mhlo_disc::SparseFillEmptyRowsOp>(
            loc, op.getOutputIndices().getType(),
            op.getOutputValues().getType(), op.getEmptyRowIndicator().getType(),
            op.getReverseIndexMap().getType(),
            RankedTensorType::get({1}, rewriter.getI64Type()), op.getIndices(),
            op.getValues(), op.getDenseShape(), op.getDefaultValue());
    // N_full is in output 4
    Value N_full_vec = sparse_fill_empty_rows_op.getResult(4);
    Value idx_zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx_one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value N_full = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(),
        rewriter.create<tensor::ExtractOp>(loc, N_full_vec, idx_zero));
    Value rank = rewriter.create<tensor::DimOp>(
        loc, sparse_fill_empty_rows_op.getResult(0), 1);
    SmallVector<int64_t, 2> output_indices_slice_shape_values,
        output_values_slice_shape_values;
    output_indices_slice_shape_values.push_back(ShapedType::kDynamic);
    // dense_shape should has static shape, we can get it's dim here
    auto output_rank =
        op.getDenseShape().getType().dyn_cast<RankedTensorType>().getDimSize(0);
    output_indices_slice_shape_values.push_back(output_rank);
    output_values_slice_shape_values.push_back(ShapedType::kDynamic);

    auto create_slice_op = [&](int slice_rank, int result_index,
                               SmallVector<int64_t, 2> slice_shape,
                               Type slice_element_type) {
      SmallVector<Value, 2> start_values(slice_rank, idx_zero);
      SmallVector<Value, 2> limit_values(slice_rank, rank);
      limit_values[0] = N_full;
      SmallVector<Value, 2> strides_values(slice_rank, idx_one);
      auto index_ty = rewriter.getIndexType();
      auto start_indices = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(start_values.size())},
                                index_ty),
          start_values);
      auto limit_indices = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(limit_values.size())},
                                index_ty),
          limit_values);
      auto strides_indices = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides_values.size())},
                                index_ty),
          strides_values);
      auto slice_op = rewriter.create<mhlo::RealDynamicSliceOp>(
          loc, RankedTensorType::get(slice_shape, slice_element_type),
          sparse_fill_empty_rows_op.getResult(result_index), start_indices,
          limit_indices, strides_indices);
      return slice_op;
    };

    auto output_indices_slice_op = create_slice_op(
        2, 0, output_indices_slice_shape_values, rewriter.getI64Type());
    auto output_values_slice_op =
        create_slice_op(1, 1, output_values_slice_shape_values, element_type);
    rewriter.replaceOp(op, {
                               output_indices_slice_op.getResult(),
                               output_values_slice_op.getResult(),
                               sparse_fill_empty_rows_op.getResult(2),
                               sparse_fill_empty_rows_op.getResult(3),
                           });
    return success();
  }
};

class ConvertSparseSegmentMeanOp
    : public OpRewritePattern<TF::SparseSegmentMeanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SparseSegmentMeanOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto reduction_mode_attr = mlir::mhlo_disc::ReductionModeEnumAttr::get(
        rewriter.getContext(), mlir::mhlo_disc::ReductionModeEnum::Mean);
    auto hlo_sparse_segment_mean =
        rewriter.create<mhlo_disc::SparseSegmentReductionOp>(
            loc, op.getOutput().getType(), op.getData(), op.getIndices(),
            op.getSegmentIds(), reduction_mode_attr);
    rewriter.replaceOp(op, hlo_sparse_segment_mean.getResult());
    return success();
  }
};

class ConvertSparseSegmentSumOp
    : public OpRewritePattern<TF::SparseSegmentSumOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::SparseSegmentSumOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto reduction_mode_attr = mlir::mhlo_disc::ReductionModeEnumAttr::get(
        rewriter.getContext(), mlir::mhlo_disc::ReductionModeEnum::Sum);
    auto hlo_sparse_segment_sum =
        rewriter.create<mhlo_disc::SparseSegmentReductionOp>(
            loc, op.getOutput().getType(), op.getData(), op.getIndices(),
            op.getSegmentIds(), reduction_mode_attr);
    rewriter.replaceOp(op, hlo_sparse_segment_sum.getResult());
    return success();
  }
};

class ConvertWhereOp : public OpRewritePattern<TF::WhereOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::WhereOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto where = rewriter.create<mhlo_disc::WhereOp>(
        loc, op.getIndex().getType(),
        RankedTensorType::get({1}, rewriter.getI64Type()), op.getInput());

    // num_output_elements
    Value num_output_elements_vec = where.getResult(1);
    Value idx_zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx_one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value num_output_elements = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(),
        rewriter.create<tensor::ExtractOp>(loc, num_output_elements_vec,
                                           idx_zero));
    Value input_dims =
        rewriter.create<tensor::DimOp>(loc, where.getResult(0), 1);

    // output is rank 2
    SmallVector<Value, 2> start_values(2, idx_zero);
    SmallVector<Value, 2> limit_values(2, input_dims);
    limit_values[0] = num_output_elements;
    SmallVector<Value, 2> strides_values(2, idx_one);
    auto index_ty = rewriter.getIndexType();
    auto create_indices = [&](SmallVector<Value, 2> indices_values) {
      auto indices = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(indices_values.size())},
                                index_ty),
          indices_values);
      return indices;
    };
    auto start_indices = create_indices(start_values);
    auto limit_indices = create_indices(limit_values);
    auto strides_indices = create_indices(strides_values);

    SmallVector<int64_t, 2> output_slice_shape_values;
    output_slice_shape_values.push_back(ShapedType::kDynamic);
    auto input_rank =
        op.getInput().getType().dyn_cast<RankedTensorType>().getRank();
    output_slice_shape_values.push_back(input_rank);

    auto index_slice_op = rewriter.create<mhlo::RealDynamicSliceOp>(
        loc,
        RankedTensorType::get(output_slice_shape_values, rewriter.getI64Type()),
        where.getResult(0), start_indices, limit_indices, strides_indices);
    rewriter.replaceOp(op, index_slice_op.getResult());
    return success();
  }
};

#include "mlir/disc/transforms/lower_tf.inc"

// add pre-defined pdll patterns here.
std::string getPredefinedPDLPatterns() {
  std::string preDefinedPatterns;

  /// Examples: FusedConvRelu
  ///
  /// static const char* fused_conv_relu = R"pdll(
  ///   Pattern TFFusedConvRelu {
  ///
  ///     /// match phase: define the pattern
  ///     let data_format_attr : Attr;
  ///     let dilations_attr : Attr;
  ///     let padding_attr : Attr;
  ///     let strides_attr : Attr;
  ///     let conv = op<tf.Conv2D>(input : Value, weights : Value) { data_format
  ///     = data_format_attr, dilations = dilations_attr, padding =
  ///     padding_attr, strides = strides_attr }; let relu =
  ///     op<tf.Relu>(conv.0);
  ///
  ///     /// rewrite phase
  ///     rewrite relu with {
  ///       /// 1. create custom call op
  ///       let inputs = PackValue_2(attr<"\"in\"">, input, weights);
  ///       let outputs = PackValue_2(attr<"\"out\"">, conv.0, relu.0);
  ///       let infos = CreateCustomCall(attr<"\"op\"">, inputs, outputs);
  ///
  ///       /// 2. set attrs that are used by bladedisc.
  ///       SetAttr(infos.op, attr<"\"call_target_name\"">,
  ///       attr<"\"disc.custom_call.fused_conv_relu\"">); SetAttr(infos.op,
  ///       attr<"\"input_placements\"">, attr<"\"h,h\"">); SetAttr(infos.op,
  ///       attr<"\"output_placements\"">, attr<"\"h\"">);
  ///
  ///       /// 3. set attrs that are directly passed to the custom call kernel.
  ///       SetCustomAttr(infos.op, attr<"\"data_format\"">, data_format_attr);
  ///       SetCustomAttr(infos.op, attr<"\"dilation\"">, dilations_attr);
  ///       SetCustomAttr(infos.op, attr<"\"padding\"">, padding_attr);
  ///       SetCustomAttr(infos.op, attr<"\"strides\"">, strides_attr);
  ///
  ///       let rs = UnpackValue_2(infos.new_outputs);
  ///       replace conv with rs.0;
  ///       replace relu with rs.1;
  ///     };
  ///   }
  /// )pdll";
  /// preDefinedPatterns += fused_conv_relu + "\n";

  return preDefinedPatterns;
}

void PrepareTFPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  if (failed(convertQuantizedConv2DWithBiasAndRequantizeOp(func))) {
    signalPassFailure();
    return;
  }

  // add pre-defined pdl patterns
  populateDiscPdlPatternsFromString(&patterns, getPredefinedPDLPatterns());

  populateDiscPdlPatternsFromFiles(&patterns, ParseFileString(pdll_files_),
                                   ParseFileString(pdll_include_dirs_));

  populateWithGenerated(patterns);
  // clang-format off
  patterns.insert<
      ConvertDequantizeOp,
      ConvertQuantizeV2Op,
      ConvertSqueezeOpDynamic,
      ConvertTopKV2OpDynamic,
      ConvertUniformOp,
      ConvertBucketizeOp,
      ConvertSparseReshapeOp,
      ConvertSparseFillEmptyRowsOp,
      ConvertSparseSegmentMeanOp,
      ConvertSparseSegmentSumOp,
      ConvertWhereOp
  >(ctx);

  if (lowerFakeQuantToQuantAndDequant()) {
    patterns.insert<ConvertFakeQuantV2Op>(ctx);
  } else {
    patterns.insert<ConvertFakeQuantOp>(ctx);
  }

  // clang-format on
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Lower some tf ops before tf2mhlo lowering.
std::unique_ptr<OperationPass<func::FuncOp>> createDiscLowerTfPass(
    const std::string& pdll_files, const std::string& pdll_include_dirs) {
  return std::make_unique<PrepareTFPass>(pdll_files, pdll_include_dirs);
}

}  // namespace disc_ral
}  // namespace mlir
