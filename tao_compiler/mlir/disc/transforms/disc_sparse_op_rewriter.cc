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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "disc-sparse-op-rewriter"

namespace mlir {
namespace disc_ral {

// sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
//     math_ops.reduce_prod(
//         array_ops.slice(original_shape, [0], [original_rank - 1])),
//     array_ops.gather(original_shape, original_rank - 1)
// ])
struct SimplifySparseReshapeOp
    : public OpRewritePattern<mhlo_disc::SparseReshapeOp> {
  using OpRewritePattern<mhlo_disc::SparseReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::SparseReshapeOp op,
                                PatternRewriter& rewriter) const override {
    // Location loc = op.getLoc();
    // Value input = op.getOperand();
    // auto input_type = input.getType().dyn_cast<RankedTensorType>();
    // if (!input_type) return failure();

    rewriter.replaceOp(op, {op.getInputIndices(), op.getInputShape()});

    return success();
  }
};

// Deal with gather generated from tf's sparse_op.sparse_retain in embedding
// column 2 mhlo.dynamic_gather should be rewrite by this OpRewritePattern 1st
// mhlo.dynamic_gather with 1-dimension input/output deal with sparse tensor's
// values 2nd mhlo.dynamic_gather with 2-dimension input/output deal with sparse
// tensor's indices After rewrite, there should be 2 real-dynamic-slice for 2 of
// the rewrited mhlo.dynamic_gather This RewritePattern is crutial for where
// ops' output fusion.
struct RewriteDynamicGatherOp : public OpRewritePattern<mhlo::DynamicGatherOp> {
  using OpRewritePattern<mhlo::DynamicGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicGatherOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getOperand();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    if (!input_type) return failure();
    auto gather_input_rank = input_type.getRank();
    if (!(gather_input_rank <= 1)) return failure();

    Value index = op.getStartIndices();
    auto index_type = index.getType().dyn_cast<RankedTensorType>();
    if (!index_type) return failure();

    auto reshape_op = index.getDefiningOp();
    if (!(reshape_op && isa<mhlo::DynamicReshapeOp>(reshape_op))) {
      return failure();
    }
    // TODO: check reshape with input (<?x1xi64>, <1xi64>) and output <?xi64>

    // find another gather that consumes reshape output
    auto reshape_users =
        llvm::to_vector<4>(reshape_op->getResult(0).getUsers());
    if (reshape_users.size() != 2) return failure();
    for (Operation* user : reshape_users) {
      if (!(user && isa<mhlo::DynamicGatherOp>(user))) {
        return failure();
      }
    }
    auto gather_1 = reshape_users[0];
    auto gather_2 = reshape_users[1];

    auto real_dynamic_slice_op = reshape_op->getOperand(0).getDefiningOp();
    if (!(real_dynamic_slice_op &&
          isa<mhlo::RealDynamicSliceOp>(real_dynamic_slice_op))) {
      return failure();
    }

    auto where_op = real_dynamic_slice_op->getOperand(0).getDefiningOp();
    if (!(where_op && isa<mhlo_disc::WhereOp>(where_op))) {
      return failure();
    }

    auto index_ty = rewriter.getIndexType();
    // 1. create a new DynamicReshapeOp op, with input 0 from mhlo_disc.where
    // also do a <?x1xi64> to <?xi64> reshape like the existing one
    auto reshape_dim_value = rewriter.create<arith::IndexCastOp>(
        loc, index_ty,
        rewriter.create<tensor::DimOp>(loc, where_op->getResult(0), 0));
    SmallVector<Value, 1> reshape_dims(1, reshape_dim_value);
    auto reshape_dim_input = rewriter.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(reshape_dims.size())},
                              index_ty),
        reshape_dims);
    auto where_output_ty =
        where_op->getResult(0).getType().dyn_cast<RankedTensorType>();
    auto oldShape = where_output_ty.getShape();
    SmallVector<int64_t, 1> newDimSizes(1, oldShape[0]);
    auto outTy =
        RankedTensorType::get(newDimSizes, where_output_ty.getElementType());
    auto new_reshape = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, outTy, where_op->getResult(0), reshape_dim_input);

    // 2. clone 2 gather op, with input from new reshape
    auto cloned_gather_op_1 = rewriter.clone(*gather_1);
    cloned_gather_op_1->setOperand(1, new_reshape->getResult(0));

    auto cloned_gather_op_2 = rewriter.clone(*gather_2);
    cloned_gather_op_2->setOperand(1, new_reshape->getResult(0));

    // 3. create dynamic slice to slice gather output
    Value idx_zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx_one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value num_output_elements = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(),
        rewriter.create<tensor::ExtractOp>(loc, where_op->getResult(1),
                                           idx_zero));

    auto create_indices = [&](SmallVector<Value, 2> indices_values) {
      auto indices = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(indices_values.size())},
                                index_ty),
          indices_values);
      return indices;
    };

    auto create_slice_op = [&](Operation* gather) {
      auto gather_input_rank = gather->getOperand(0)
                                   .getType()
                                   .dyn_cast<RankedTensorType>()
                                   .getRank();
      Value gather_output = gather->getResult(0);

      SmallVector<Value, 2> start_values(gather_input_rank, idx_zero);
      SmallVector<Value, 2> limit_values(gather_input_rank,
                                         num_output_elements);
      SmallVector<Value, 2> strides_values(gather_input_rank, idx_one);
      if (gather_input_rank == 2) {
        limit_values[1] = rewriter.create<tensor::DimOp>(loc, gather_output, 1);
      }

      auto start_indices = create_indices(start_values);
      auto limit_indices = create_indices(limit_values);
      auto strides_indices = create_indices(strides_values);

      SmallVector<int64_t, 2> output_slice_shape_values(gather_input_rank, -1);
      auto slice_input_shape =
          gather_output.getType().dyn_cast<RankedTensorType>().getShape();
      if (gather_input_rank == 2) {
        output_slice_shape_values[1] = slice_input_shape[1];
      }
      auto gather_slice_op = rewriter.create<mhlo::RealDynamicSliceOp>(
          loc,
          RankedTensorType::get(output_slice_shape_values,
                                rewriter.getI64Type()),
          gather_output, start_indices, limit_indices, strides_indices);
      return gather_slice_op;
    };

    auto gather_1_slice = create_slice_op(cloned_gather_op_1);
    auto gather_2_slice = create_slice_op(cloned_gather_op_2);

    rewriter.replaceOp(gather_1, gather_1_slice->getResult(0));
    rewriter.replaceOp(gather_2, gather_2_slice->getResult(0));

    return success();
  }
};

struct RewriteSparseSegmentReductionOp
    : public OpRewritePattern<mhlo_disc::SparseSegmentReductionOp> {
  using OpRewritePattern<mhlo_disc::SparseSegmentReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::SparseSegmentReductionOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    Value indices = op.getIndices();
    auto indices_type = indices.getType().dyn_cast<RankedTensorType>();
    if (!indices_type) return failure();

    auto segment_ids = op.getSegmentIds();
    auto segment_ids_type = segment_ids.getType().dyn_cast<RankedTensorType>();
    if (!segment_ids_type) return failure();

    auto output = op.getOutput();
    auto output_type = output.getType().dyn_cast<RankedTensorType>();
    if (!output_type) return failure();

    // check sparse_fill_empty_rows:1 --> real_dynamic_slice --> indices
    auto real_dynamic_slice_1 = indices.getDefiningOp();
    if (!(real_dynamic_slice_1 &&
          isa<mhlo::RealDynamicSliceOp>(real_dynamic_slice_1))) {
      return failure();
    }

    auto sparse_fill_empty_rows =
        real_dynamic_slice_1->getOperand(0).getDefiningOp();
    if (!(sparse_fill_empty_rows &&
          isa<mhlo_disc::SparseFillEmptyRowsOp>(sparse_fill_empty_rows))) {
      return failure();
    }

    // check sparse_fill_empty_rows:0 --> real_dynamic_slice -->
    // real_dynamic_slice --> dynamic_reshape
    auto dynamic_reshape = segment_ids.getDefiningOp();
    if (!(dynamic_reshape && isa<mhlo::DynamicReshapeOp>(dynamic_reshape))) {
      return failure();
    }

    // TODO(lanbo): check real dynamic slice is doing operations like ids[:, 0]
    auto real_dynamic_slice_2 = dynamic_reshape->getOperand(0).getDefiningOp();
    if (!(real_dynamic_slice_2 &&
          isa<mhlo::RealDynamicSliceOp>(real_dynamic_slice_2))) {
      return failure();
    }

    auto real_dynamic_slice_3 =
        real_dynamic_slice_2->getOperand(0).getDefiningOp();
    if (!(real_dynamic_slice_3 &&
          isa<mhlo::RealDynamicSliceOp>(real_dynamic_slice_3))) {
      return failure();
    }

    auto sparse_fill_empty_rows_1 =
        real_dynamic_slice_3->getOperand(0).getDefiningOp();
    if (!(sparse_fill_empty_rows &&
          isa<mhlo_disc::SparseFillEmptyRowsOp>(sparse_fill_empty_rows))) {
      return failure();
    }
    // (TODO) check from same sparse_fill_empty_rows
    // if (sparse_fill_empty_rows->getOperation() !=
    //     sparse_fill_empty_rows_1->getOperation()) {
    //   return failure();
    // }
    // get input and output from sparse_fill_empty_rows
    auto sparse_fill_op = dyn_cast_or_null<mhlo_disc::SparseFillEmptyRowsOp>(
        sparse_fill_empty_rows);
    auto origin_indices = sparse_fill_op.getIndices();
    auto origin_values = sparse_fill_op.getValues();
    auto dense_shape = sparse_fill_op.getDenseShape();
    auto default_value = sparse_fill_op.getDefaultValue();
    auto empty_row_indicator = sparse_fill_op.getEmptyRowIndicator();
    auto empty_row_indicator_type =
        empty_row_indicator.getType().dyn_cast<RankedTensorType>();
    if (!empty_row_indicator_type) return failure();

    op.getOperation()->dump();
#if 1
    auto empty_row_users = llvm::to_vector<4>(empty_row_indicator.getUsers());
    if (empty_row_users.size() > 1) {
      rewriter.setInsertionPointAfter(sparse_fill_empty_rows);
    } else {
      rewriter.setInsertionPoint(sparse_fill_empty_rows);
    }
#else
    rewriter.setInsertionPoint(sparse_fill_empty_rows);
#endif
    auto sparse_segment_reduction_with_empty_rows =
        rewriter.create<mhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(
            loc, output_type, empty_row_indicator_type, op.getData(),
            origin_values, origin_indices, dense_shape,
            rewriter.getBoolAttr(op.getIsMean()));

    // redirect use of origin empty_row_indicator to newly created op's output 1
    // TODO(lanbo.llb): redirect reverse index map use
    empty_row_indicator.replaceAllUsesWith(
        sparse_segment_reduction_with_empty_rows.getResult(1));
    rewriter.replaceOp(op,
                       sparse_segment_reduction_with_empty_rows.getResult(0));
    return success();
  }
};

struct DiscSparseOpRewriterPass
    : public DiscSparseOpRewriterPassBase<DiscSparseOpRewriterPass> {
  void runOnOperation() override;
};

bool initSparseSegmentReductionRewrite() {
  const char* env = getenv("DISC_ENABLE_SPARSE_SEGMENT_REDUCTION_REWRITE");
  if (!env) return true;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

void DiscSparseOpRewriterPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  patterns.insert<SimplifySparseReshapeOp>(patterns.getContext());
  patterns.insert<RewriteDynamicGatherOp>(patterns.getContext());
  if (initSparseSegmentReductionRewrite()) {
    patterns.insert<RewriteSparseSegmentReductionOp>(patterns.getContext());
  }

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createDiscSparseOpRewriterPass() {
  return std::make_unique<DiscSparseOpRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir
