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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"                          // TF:llvm-project
#include "mlir/IR/Location.h"                            // TF:llvm-project
#include "mlir/IR/MLIRContext.h"                         // TF:llvm-project
#include "mlir/IR/Operation.h"                           // TF:llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // TF:llvm-project
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"

#define DEBUG_TYPE "conv-rewriter"

namespace mlir {
namespace disc_ral {

namespace {

// - input layput: each field for one dimension. The order is:
//   * batch, channel, spatial dimensions
// - kernel layout: each field for one dimension. The order is:
//   * in_channel, out_channel, spatial dimensions
// - output layout: each field for one dimension. The order is:
//   * batch, channel, spatial dimensions
template <typename T>
struct ConvParams {
  int num_spatial_dims;
  Value input;
  Value filter;
  Value output;
  SmallVector<int64_t> inputLayout;
  SmallVector<int64_t> filterLayout;
  SmallVector<int64_t> outputLayout;
  SmallVector<int64_t> expectedInputLayout;
  SmallVector<int64_t> expectedFilterLayout;
  SmallVector<int64_t> expectedOutputLayout;
  T conv;
};

template <typename T>
LogicalResult fillConvParams(T op, ConvParams<T>& params) {
  auto inputTy = params.input.getType().template dyn_cast<RankedTensorType>();
  auto filterTy = params.filter.getType().template dyn_cast<RankedTensorType>();
  auto paddingTy =
      op.getDPadding().getType().template dyn_cast<RankedTensorType>();
  auto outputTy = params.output.getType().template dyn_cast<RankedTensorType>();

  if (!inputTy || !filterTy || !paddingTy || !outputTy) {
    return op.emitOpError() << "operands must be ranked type";
  }

  int rank = filterTy.getRank();
  params.num_spatial_dims = rank - 2;

  if (params.num_spatial_dims < 1) {
    return op.emitOpError() << "conv op's input rank is less than 3";
  }

  auto dimension_numbers = op.getDimensionNumbers();
  params.inputLayout.push_back(dimension_numbers.getInputBatchDimension());
  params.inputLayout.push_back(dimension_numbers.getInputFeatureDimension());
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  for (int i = 0; i < params.num_spatial_dims; ++i) {
    params.inputLayout.push_back(input_spatial_dimensions[i]);
  }

  params.filterLayout.push_back(
      dimension_numbers.getKernelInputFeatureDimension());
  params.filterLayout.push_back(
      dimension_numbers.getKernelOutputFeatureDimension());
  auto filter_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  for (int i = 0; i < params.num_spatial_dims; ++i) {
    params.filterLayout.push_back(filter_spatial_dimensions[i]);
  }

  params.outputLayout.push_back(dimension_numbers.getOutputBatchDimension());
  params.outputLayout.push_back(dimension_numbers.getOutputFeatureDimension());
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  for (int i = 0; i < params.num_spatial_dims; ++i) {
    params.outputLayout.push_back(output_spatial_dimensions[i]);
  }
  return success();
}

LogicalResult extractConvParams(mhlo::DynamicConvOp op,
                                ConvParams<mhlo::DynamicConvOp>& params) {
  params.conv = op;
  params.input = op.getLhs();
  params.filter = op.getRhs();
  params.output = op.getResult();

  return fillConvParams<mhlo::DynamicConvOp>(op, params);
}

LogicalResult extractConvParams(
    mlir::mhlo_disc::QuantizedDynamicConvOp op,
    ConvParams<mlir::mhlo_disc::QuantizedDynamicConvOp>& params) {
  params.conv = op;
  params.input = op.getInput();
  params.filter = op.getWeight();
  params.output = op.getResult();

  return fillConvParams<mlir::mhlo_disc::QuantizedDynamicConvOp>(op, params);
}

void fillNCHW(SmallVector<int64_t>& layout, int num_spatial_dims) {
  layout[0] = 0;
  layout[1] = 1;
  for (int i = 0; i < num_spatial_dims; ++i) {
    layout[2 + i] = 2 + i;
  }
}

void fillNHWC(SmallVector<int64_t>& layout, int num_spatial_dims) {
  layout[0] = 0;
  layout[1] = num_spatial_dims + 1;
  for (int i = 0; i < num_spatial_dims; ++i) {
    layout[2 + i] = 1 + i;
  }
}

void fillOIHW(SmallVector<int64_t>& layout, int num_spatial_dims) {
  layout[0] = 1;
  layout[1] = 0;
  for (int i = 0; i < num_spatial_dims; ++i) {
    layout[2 + i] = 2 + i;
  }
}

void fillHWIO(SmallVector<int64_t>& layout, int num_spatial_dims) {
  layout[0] = num_spatial_dims;
  layout[1] = num_spatial_dims + 1;
  for (int i = 0; i < num_spatial_dims; ++i) {
    layout[2 + i] = i;
  }
}

void fillOHWI(SmallVector<int64_t>& layout, int num_spatial_dims) {
  layout[0] = num_spatial_dims + 1;
  layout[1] = 0;
  for (int i = 0; i < num_spatial_dims; ++i) {
    layout[2 + i] = i + 1;
  }
}

SmallVector<int64_t> inferTransposeAttr(
    const SmallVector<int64_t>& layout,
    const SmallVector<int64_t>& expectedLayout) {
  SmallVector<int64_t> transposeAttr(layout.size());
  for (size_t i = 0; i < layout.size(); ++i) {
    transposeAttr[expectedLayout[i]] = layout[i];
  }
  return transposeAttr;
}

inline DenseIntElementsAttr ConvertIntVecToDenseIntElementsAttr(
    llvm::ArrayRef<int64_t> op_dimensions, PatternRewriter& rewriter) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(op_dimensions.size(), rewriter.getIntegerType(64)),
      op_dimensions);
}

struct ConvToDynamicConvConvert : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getLhs();
    Value filter = op.getRhs();

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

    auto i64Vec = ConvertDenseIntAttr(op.getPadding());
    std::vector<int32_t> paddingValues(i64Vec.size());
    std::transform(
        i64Vec.begin(), i64Vec.end(), paddingValues.begin(),
        [](int64_t val) -> int32_t { return static_cast<int32_t>(val); });

    RankedTensorType ty = RankedTensorType::get({paddingValues.size()},
                                                rewriter.getIntegerType(32));
    Value paddingTensor = rewriter.create<mhlo::ConstantOp>(
        loc, mlir::DenseIntElementsAttr::get(ty, paddingValues));

    SmallVector<Value, 4> newOperands = {input, filter, paddingTensor};

    if (op->getAttr("padding")) op->removeAttr("padding");

    Value dynamicConv = rewriter.replaceOpWithNewOp<mhlo::DynamicConvOp>(
        op, op.getType(), newOperands, op->getAttrs());

    return success();
  }
};

// The basic logic of this pass is to insert transpose op to ensure the conv op
// having expected format.
template <typename T>
struct DiscConvRewriterPass
    : public ConvRewriterPassBase<DiscConvRewriterPass<T>> {
  explicit DiscConvRewriterPass(int cc_major, int cc_minor)
      : ConvRewriterPassBase<DiscConvRewriterPass<T>>::ConvRewriterPassBase() {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }

  // Returns true if the conv is depthwise.
  bool isDepthwiseConv(ConvParams<T>& params) {
    auto filterTy = params.filter.getType().template cast<RankedTensorType>();
    return filterTy.getShape()[params.filterLayout[0]] == 1;
  }

  LogicalResult inferExpectedLayout(ConvParams<T>& params, int cc_major) {
    bool onGpu = placement_utils::isGpuMhlo(params.conv);
    auto inputTy = params.input.getType().template dyn_cast<RankedTensorType>();
    auto filterTy =
        params.filter.getType().template dyn_cast<RankedTensorType>();
    auto outputTy =
        params.output.getType().template dyn_cast<RankedTensorType>();
    int rank = inputTy.getRank();
    int num_spatial_dims = params.num_spatial_dims;

    auto& inputLayout = params.expectedInputLayout;
    auto& filterLayout = params.expectedFilterLayout;
    auto& outputLayout = params.expectedOutputLayout;
    inputLayout.resize(rank);
    filterLayout.resize(rank);
    outputLayout.resize(rank);

    if (onGpu) {
      bool is_fp32 =
          inputTy.getElementType().isF32() && filterTy.getElementType().isF32();
      bool use_tf32 = true;
      TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("NVIDIA_TF32_OVERRIDE",
                                                 /*default_val=*/use_tf32,
                                                 &use_tf32));
#if !defined(TENSORFLOW_USE_ROCM)
      if (cc_major >= 8 && (!is_fp32 || use_tf32) ||
          inputTy.getElementType().isF16() &&
              filterTy.getElementType().isF16()) {
        // TensorCore prefers NHWC layouts
        fillNHWC(inputLayout, num_spatial_dims);
        fillNHWC(outputLayout, num_spatial_dims);
        fillOHWI(filterLayout, num_spatial_dims);
      } else {
        // Default is NCHW & OIHW
        fillNCHW(inputLayout, num_spatial_dims);
        fillNCHW(outputLayout, num_spatial_dims);
        fillOIHW(filterLayout, num_spatial_dims);
      }
#else
      fillNCHW(inputLayout, num_spatial_dims);
      fillNCHW(outputLayout, num_spatial_dims);
      fillOIHW(filterLayout, num_spatial_dims);
#endif
    } else {
#if defined(TAO_AARCH64)
      if (isDepthwiseConv(params)) {
        fillNHWC(inputLayout, num_spatial_dims);
        fillNHWC(outputLayout, num_spatial_dims);
        fillHWIO(filterLayout, num_spatial_dims);
      } else {
        fillNHWC(inputLayout, num_spatial_dims);
        fillNHWC(outputLayout, num_spatial_dims);
        fillOHWI(filterLayout, num_spatial_dims);
      }
#else
      // CPU conv, default layout:
      fillNHWC(inputLayout, num_spatial_dims);
      fillNHWC(outputLayout, num_spatial_dims);
      fillHWIO(filterLayout, num_spatial_dims);
#endif
    }

    return success();
  }

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
      T op, Value value, const SmallVectorImpl<int64_t>& transpose_permutation,
      OpBuilder& b) {
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

    if (auto attr = op->getAttr(placement_utils::kDiscPlaceAssignment))
      transpose_op->setAttr(placement_utils::kDiscPlaceAssignment, attr);

    return transpose_op;
  }

  void MaybeRewriteInputFormat(ConvParams<T>& params) {
    if (params.inputLayout == params.expectedInputLayout) return;

    auto transposeAttr =
        inferTransposeAttr(params.inputLayout, params.expectedInputLayout);

    OpBuilder b(params.conv);
    auto transpose_op =
        InsertTranspose(params.conv, params.input, transposeAttr, b);
    params.conv.getOperation()->setOperand(0, transpose_op->getResult(0));
  }

  void MaybeRewriteFilterFormat(ConvParams<T>& params) {
    if (params.filterLayout == params.expectedFilterLayout) return;

    auto transposeAttr =
        inferTransposeAttr(params.filterLayout, params.expectedFilterLayout);

    OpBuilder b(params.conv);
    auto transpose_op =
        InsertTranspose(params.conv, params.filter, transposeAttr, b);
    params.conv.getOperation()->setOperand(1, transpose_op->getResult(0));
  }

  void MaybeRewriteOutputFormat(ConvParams<T>& params) {
    if (params.outputLayout == params.expectedOutputLayout) return;

    auto in2OutAttr =
        inferTransposeAttr(params.outputLayout, params.expectedOutputLayout);
    auto out2InAttr =
        inferTransposeAttr(params.expectedOutputLayout, params.outputLayout);

    OpBuilder b(params.conv);
    b.setInsertionPointAfter(params.conv);
    auto new_tp = GetTransposeOutputType(params.output, in2OutAttr, b);
    auto originalOutputTy = params.output.getType();
    params.output.setType(new_tp);

    auto transpose_op =
        InsertTranspose(params.conv, params.output, out2InAttr, b);
    transpose_op->getResult(0).setType(originalOutputTy);
    params.output.replaceAllUsesWith(transpose_op->getResult(0));
    transpose_op->setOperand(0, params.output);
  }

  void UpdateAttributes(ConvParams<T>& params) {
    OpBuilder b(params.conv);
    SmallVector<int64_t, 2> inputSpatialDims;
    SmallVector<int64_t, 2> filterSpatialDims;
    SmallVector<int64_t, 2> outputSpatialDims;
    for (int i = 0; i < params.num_spatial_dims; ++i) {
      inputSpatialDims.push_back(params.expectedInputLayout[i + 2]);
      filterSpatialDims.push_back(params.expectedFilterLayout[i + 2]);
      outputSpatialDims.push_back(params.expectedOutputLayout[i + 2]);
    }

    params.conv.getOperation()->setAttr(
        "dimension_numbers",
        mlir::mhlo::ConvDimensionNumbersAttr::get(
            b.getContext(),
            /*inputBatchDimension*/ params.expectedInputLayout[0],
            /*inputFeatureDimension*/ params.expectedInputLayout[1],
            /*inputSpatialDimensions*/ inputSpatialDims,
            /*kernelInputFeatureDimension*/ params.expectedFilterLayout[0],
            /*kernelOutputFeatureDimension*/ params.expectedFilterLayout[1],
            /*kernelSpatialDimensions*/ filterSpatialDims,
            /*outputBatchDimension*/ params.expectedOutputLayout[0],
            /*outputFeatureDimension*/ params.expectedOutputLayout[1],
            /*outputSpatialDimensions*/ outputSpatialDims));
  }

  LogicalResult RewriteOp(T op) {
    ConvParams<T> params;
    if (failed(extractConvParams(op, params))) {
      return failure();
    }

    if (failed(inferExpectedLayout(params, cc_major_))) {
      return failure();
    }

    MaybeRewriteInputFormat(params);
    MaybeRewriteFilterFormat(params);
    MaybeRewriteOutputFormat(params);
    UpdateAttributes(params);

    return success();
  }

  LogicalResult convToDynamicConv() {
    func::FuncOp func = this->getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvToDynamicConvConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    if (std::is_same<T, mhlo::DynamicConvOp>::value) {
      if (failed(convToDynamicConv())) {
        this->signalPassFailure();
        return;
      }
    }

    SmallVector<T, 4> ops;
    this->getOperation().walk([&](T op) { ops.push_back(op); });

    // TODO(disc): We rewrite each conv op seperately, thus may lead to
    // unnecessary transpose ops. We may implement another layout optimize pass
    // in case necessary.
    for (auto& op : ops) {
      if (failed(RewriteOp(op))) {
        this->signalPassFailure();
        return;
      }
    }
  }

 private:
  int cc_major_;
  int cc_minor_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscConvRewriter(
    int cc_major, int cc_minor) {
  return std::make_unique<DiscConvRewriterPass<mhlo::DynamicConvOp>>(cc_major,
                                                                     cc_minor);
}

std::unique_ptr<OperationPass<func::FuncOp>> createDiscQuantizedConvRewriter(
    int cc_major, int cc_minor) {
  return std::make_unique<
      DiscConvRewriterPass<mlir::mhlo_disc::QuantizedDynamicConvOp>>(cc_major,
                                                                     cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir
