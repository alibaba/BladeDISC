/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file canonicalize conv ops in hlo dialect to match the
// format of CUDNN library call.
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // TF:llvm-project
#include "mlir/IR/Attributes.h"                          // TF:llvm-project
#include "mlir/IR/Location.h"                            // TF:llvm-project
#include "mlir/IR/MLIRContext.h"                         // TF:llvm-project
#include "mlir/IR/Operation.h"                           // TF:llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "transforms/PassDetail.h"

#define DEBUG_TYPE "conv-rewriter"

namespace mlir {
namespace disc_ral {

namespace {

inline DenseIntElementsAttr ConvertIntVecToDenseIntElementsAttr(
    llvm::ArrayRef<int64_t> op_dimensions, PatternRewriter& rewriter) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(op_dimensions.size(), rewriter.getIntegerType(64)),
      op_dimensions);
}

struct ConvToDynamicConvConvert : public OpRewritePattern<mhlo::ConvOp> {
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.lhs();
    Value filter = op.rhs();

    auto input_tp = input.getType().dyn_cast<RankedTensorType>();
    auto filter_tp = filter.getType().dyn_cast<RankedTensorType>();

    if (!input_tp || !filter_tp) {
      LLVM_DEBUG(llvm::dbgs() << "only ranked inputs are supported.\n");
      return failure();
    }

    int rank = input_tp.getRank();
    if (rank <= 2) {
      LLVM_DEBUG(llvm::dbgs() << "rank of input should be larger than 2.\n");
      return failure();
    }

    auto i64Vec = ConvertDenseIntAttr(op.padding());
    std::vector<int32_t> paddingValues(i64Vec.size());
    std::transform(
        i64Vec.begin(), i64Vec.end(), paddingValues.begin(),
        [](int64_t val) -> int32_t { return static_cast<int32_t>(val); });

    RankedTensorType ty = RankedTensorType::get({paddingValues.size()},
                                                rewriter.getIntegerType(32));
    Value paddingTensor = rewriter.create<mhlo::ConstOp>(
        loc, mlir::DenseIntElementsAttr::get(ty, paddingValues));

    SmallVector<Value, 4> newOperands = {input, filter, paddingTensor};

    if (op->getAttr("padding")) op->removeAttr("padding");

    Value dynamicConv = rewriter.replaceOpWithNewOp<mhlo::DynamicConvOp>(
        op, op.getType(), newOperands, op->getAttrs());

    return success();
  }
};

// # Convolution Forword Op
//   output = DConv(input, filter, ...)
// ## supported input format
//    - NCHW (preferred on GPU?)
//    - NHWC
// ## supported filter (kernel) format
//    - OIHW (preferred on GPU?)
//    - OHWI
// ## supported output format
//    - NCHW (preferred on GPU?)
//    - NHWC
// The basic logic of this pass is to insert transpose op to ensure the conv op
// having cudnn-friendly format.
struct DiscConvRewriterPass
    : public ConvRewriterPassBase<DiscConvRewriterPass> {
  explicit DiscConvRewriterPass()
      : ConvRewriterPassBase<DiscConvRewriterPass>::ConvRewriterPassBase() {}

  int rank;
  int num_spatial_dims;
  Value input;
  Value filter;
  Value padding;
  Value output;

  RankedTensorType GetTransposeOutputType(
      Value value, const SmallVectorImpl<int64_t>& transpose_permutation,
      OpBuilder& b) {
    // Compute the resulting shape.
    llvm::SmallVector<int64_t, 4> transposed_shape;
    ShapedType input_type = value.getType().cast<ShapedType>();
    auto input_shape = input_type.getShape();
    for (int64_t val : transpose_permutation) {
      transposed_shape.push_back(input_shape[val]);
    }
    return RankedTensorType::get(transposed_shape, input_type.getElementType());
  }

  Operation* InsertTranspose(
      mhlo::DynamicConvOp op, Value value,
      const SmallVectorImpl<int64_t>& transpose_permutation, OpBuilder& b) {
    auto transpose_permutation_attr =
        GetI64ElementsAttr(transpose_permutation, &b);

    // Compute the resulting shape.
    llvm::SmallVector<int64_t, 4> transposed_shape;
    ShapedType input_type = value.getType().cast<ShapedType>();
    auto input_shape = input_type.getShape();
    for (auto val : transpose_permutation) {
      transposed_shape.push_back(input_shape[val]);
    }
    auto transpose_type =
        GetTransposeOutputType(value, transpose_permutation, b);
    auto transpose_op = b.create<mhlo::TransposeOp>(
        op.getLoc(), transpose_type, value, transpose_permutation_attr);

    return transpose_op;
  }

  void MaybeRewriteInputFormat(mhlo::DynamicConvOp op) {
    // convert input format to NCHW unconditionally ATM. Re-visit this for CPU.
    auto dimension_numbers = op.dimension_numbers();
    int64_t input_batch_dimension =
        dimension_numbers.input_batch_dimension().getInt();
    int64_t input_feature_dimension =
        dimension_numbers.input_feature_dimension().getInt();
    auto input_spatial_dimensions =
        ConvertDenseIntAttr(dimension_numbers.input_spatial_dimensions());

    SmallVector<int64_t, 4> dst_format = {0, 1};
    SmallVector<int64_t, 4> src_format = {input_batch_dimension,
                                          input_feature_dimension};
    for (int i = 0; i < num_spatial_dims; ++i) {
      dst_format.push_back(i + 2);
      src_format.push_back(input_spatial_dimensions[i]);
    }

    if (dst_format == src_format) {
      return;
    }

    OpBuilder b(op);
    auto transpose_op = InsertTranspose(op, input, src_format, b);
    op.getOperation()->setOperand(0, transpose_op->getResult(0));
  }

  void MaybeRewriteFilterFormat(mhlo::DynamicConvOp op) {
    // convert filter format to OIHW unconditionally ATM. Re-visit this for CPU.
    auto dimension_numbers = op.dimension_numbers();
    int64_t filter_input_feature_dimension =
        dimension_numbers.kernel_input_feature_dimension().getInt();
    int64_t filter_output_feature_dimension =
        dimension_numbers.kernel_output_feature_dimension().getInt();
    auto filter_spatial_dimensions =
        ConvertDenseIntAttr(dimension_numbers.kernel_spatial_dimensions());

    SmallVector<int64_t, 4> dst_format = {0, 1};
    SmallVector<int64_t, 4> src_format = {filter_output_feature_dimension,
                                          filter_input_feature_dimension};
    for (int i = 0; i < num_spatial_dims; ++i) {
      dst_format.push_back(i + 2);
      src_format.push_back(filter_spatial_dimensions[i]);
    }

    if (dst_format == src_format) {
      return;
    }

    OpBuilder b(op);
    auto transpose_op = InsertTranspose(op, filter, src_format, b);
    op.getOperation()->setOperand(1, transpose_op->getResult(0));
  }

  void MaybeRewriteOutputFormat(mhlo::DynamicConvOp op) {
    // convert output format to NCHW unconditionally ATM. Re-visit this for CPU.
    auto dimension_numbers = op.dimension_numbers();
    int64_t output_batch_dimension =
        dimension_numbers.output_batch_dimension().getInt();
    int64_t output_feature_dimension =
        dimension_numbers.output_feature_dimension().getInt();
    auto output_spatial_dimensions =
        ConvertDenseIntAttr(dimension_numbers.output_spatial_dimensions());

    SmallVector<int64_t, 4> dst_format = {0, 1};
    SmallVector<int64_t, 4> src_format = {output_batch_dimension,
                                          output_feature_dimension};
    for (int i = 0; i < num_spatial_dims; ++i) {
      dst_format.push_back(i + 2);
      src_format.push_back(output_spatial_dimensions[i]);
    }

    if (dst_format == src_format) {
      return;
    }

    OpBuilder b(op);
    b.setInsertionPointAfter(op);
    auto new_tp = GetTransposeOutputType(output, src_format, b);
    output.setType(new_tp);

    SmallVector<int64_t, 4> transpose_permutation(rank, 0);
    for (int i = 0; i < rank; ++i) {
      transpose_permutation[src_format[i]] = dst_format[i];
    }

    auto transpose_op = InsertTranspose(op, output, transpose_permutation, b);
    output.replaceAllUsesWith(transpose_op->getResult(0));
    transpose_op->setOperand(0, output);
  }

  void UpdateAttributes(mhlo::DynamicConvOp op) {
    OpBuilder b(op);
    SmallVector<int64_t, 2> spatial_dims;
    for (int i = 0; i < num_spatial_dims; ++i) {
      spatial_dims.push_back(i + 2);
    }

    // dst input/output format is always NCHW ATM.
    IntegerAttr batch_dim_attr = b.getI64IntegerAttr(0);
    IntegerAttr feature_dim_attr = b.getI64IntegerAttr(1);
    DenseIntElementsAttr spatial_dims_attr =
        GetI64ElementsAttr(spatial_dims, &b);

    // dst filter format is always OIHW ATM.
    IntegerAttr kernel_input_feature_dim_attr = b.getI64IntegerAttr(1);
    IntegerAttr kernel_output_feature_dim_attr = b.getI64IntegerAttr(0);

    op.getOperation()->setAttr(
        "dimension_numbers",
        mlir::mhlo::ConvDimensionNumbers::get(
            batch_dim_attr, feature_dim_attr, spatial_dims_attr,
            kernel_input_feature_dim_attr, kernel_output_feature_dim_attr,
            spatial_dims_attr, batch_dim_attr, feature_dim_attr,
            spatial_dims_attr, b.getContext()));
  }

  void RewriteOp(mhlo::DynamicConvOp op) {
    input = op.lhs();
    filter = op.rhs();
    padding = op.d_padding();
    output = op.getResult();

    auto input_tp = input.getType().dyn_cast<RankedTensorType>();
    auto filter_tp = filter.getType().dyn_cast<RankedTensorType>();
    auto padding_tp = padding.getType().dyn_cast<RankedTensorType>();

    if (!input_tp || !filter_tp || !padding_tp) {
      op.emitOpError() << "operands must be ranked type";
      return;
    }

    Location loc = op.getLoc();
    rank = filter_tp.getRank();
    num_spatial_dims = rank - 2;

    if (num_spatial_dims < 1) {
      op.emitOpError() << "conv op's input rank is less than 3";
      return;
    }

    // We only support Conv2D ATM.
    if (num_spatial_dims != 2) {
      return;
    }

    MaybeRewriteInputFormat(op);
    MaybeRewriteFilterFormat(op);
    MaybeRewriteOutputFormat(op);

    UpdateAttributes(op);
  }

  LogicalResult convToDynamicConv() {
    FuncOp func = getFunction();
    MLIRContext* ctx = func.getContext();
    OwningRewritePatternList patterns(ctx);
    patterns.insert<ConvToDynamicConvConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      return failure();
    }
    return success();
  }

  void runOnFunction() override {
    if (failed(convToDynamicConv())) {
      signalPassFailure();
      return;
    }

    SmallVector<mhlo::DynamicConvOp, 4> ops;
    getFunction().walk([&](mhlo::DynamicConvOp op) { ops.push_back(op); });

    // TODO(disc): We rewrite each conv op seperately, thus may lead to
    // unnecessary transpose ops. We may implement another layout optimize pass
    // in case necessary.
    for (auto& op : ops) {
      RewriteOp(op);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscConvRewriter() {
  return std::make_unique<DiscConvRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir