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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "disc-dynamic-slice-converter"

// This file implements conversion from mhlo.dynamic_slice to
// mhlo.real_dynamic_slice. This conversion only works for mhlo.dynamic_slice
// with dynamic shape input. mhlo.dynamic_slice with static shape input can be
// canonicalized by it's own canonicalization logic.

namespace mlir {
namespace disc_ral {

// Clamps value to the range [lower, upper].  Requires lower <= upper.
template <typename T>
static T clamp(const T& value, const T& lower, const T& upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

struct ConvertDynamicSliceOp : public OpRewritePattern<mhlo::DynamicSliceOp> {
  using OpRewritePattern<mhlo::DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    if (!input_type || input_type.hasStaticShape()) return failure();

    // slice_sizes: i64 attr
    auto slice_sizes = op.getSliceSizes().getValues<int64_t>();
    SmallVector<int64_t, 4> slice_size;
    for (const auto& size : slice_sizes) {
      slice_size.push_back(size);
    }

    SmallVector<Value, 4> start_indices_values, limit_indices_values;
    Value lower_bound =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    Value one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    // start_indices is variadic
    for (const auto& [index, value] : llvm::enumerate(op.getStartIndices())) {
      APInt val;
      // if constant value, make a arith.constant
      bool if_const_val = matchPattern(value, m_ConstantInt(&val));
      Value clamped_start_value;
      // konwn dim
      if (if_const_val && input_type.getDimSize(index) != -1) {
        int64_t clampedStart =
            clamp(val.getSExtValue(), static_cast<int64_t>(0),
                  input_type.getDimSize(index) - slice_sizes[index]);
        clamped_start_value = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(clampedStart));
        start_indices_values.push_back(clamped_start_value);
      } else {
        Value dim_size_casted = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(),
            rewriter.create<tensor::DimOp>(loc, input, index));
        Value slice_size_value = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(slice_sizes[index]));
        Value upper_bound = rewriter.create<arith::SubIOp>(loc, dim_size_casted,
                                                           slice_size_value);
        if (if_const_val) {
          Value val_const = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(val.getSExtValue()));
          clamped_start_value = rewriter.create<arith::MinSIOp>(
              loc, rewriter.create<arith::MaxSIOp>(loc, lower_bound, val_const),
              upper_bound);
        } else {
          Value start_val = rewriter.create<arith::ExtSIOp>(
              loc, rewriter.getI64Type(),
              rewriter.create<tensor::ExtractOp>(loc, value));
          clamped_start_value = rewriter.create<arith::MinSIOp>(
              loc, rewriter.create<arith::MaxSIOp>(loc, lower_bound, start_val),
              upper_bound);
        }
        start_indices_values.push_back(clamped_start_value);
      }
      Value limit = rewriter.create<arith::AddIOp>(
          loc, clamped_start_value,
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(slice_sizes[index])));
      limit_indices_values.push_back(limit);
    }
    SmallVector<Value, 4> strides_indices_values(start_indices_values.size(),
                                                 one);
    auto index_ty = rewriter.getI64Type();
    auto start_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get(
            {static_cast<int64_t>(start_indices_values.size())}, index_ty),
        start_indices_values);
    auto limit_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get(
            {static_cast<int64_t>(limit_indices_values.size())}, index_ty),
        limit_indices_values);
    auto strides_indices = rewriter.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get(
            {static_cast<int64_t>(strides_indices_values.size())}, index_ty),
        strides_indices_values);
    auto slice_op = rewriter.create<mhlo::RealDynamicSliceOp>(
        loc, RankedTensorType::get(slice_size, input_type.getElementType()),
        input, start_indices, limit_indices, strides_indices);

    rewriter.replaceOp(op, slice_op.getResult());
    return success();
  }
};

struct DiscDynamicSliceConverterPass
    : public DiscDynamicSliceConverterPassBase<DiscDynamicSliceConverterPass> {
  void runOnOperation() override;
};

void DiscDynamicSliceConverterPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  patterns.insert<ConvertDynamicSliceOp>(patterns.getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDynamicSliceConverterPass() {
  return std::make_unique<DiscDynamicSliceConverterPass>();
}

}  // namespace disc_ral
}  // namespace mlir
