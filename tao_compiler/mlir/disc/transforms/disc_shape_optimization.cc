/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

// This file implements the logic to do some shape optimizations on tensor
// level.
#include <chrono>
#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"

#define DEBUG_TYPE "disc-shape-optimization"

#define DISC_DEBUG(x) LLVM_DEBUG(x)

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;

namespace {

/////////////////////////// Stage #1 BEGIN ////////////////////////////////////

// Insert disc_shape.tie_shape for each value having RankedTensorType type.
// before
//  ```
//    %0 = ... : tensor<?x10xf32>
//    use(%0)
//  ```
// after
//  ```
//    %0 = ... : tensor<?x10xf32>
//    %0_d0 = tensor.dim %0, %c0 : tensor<?x10xf32>
//    %0_d1 = tensor.dim %0, %c1 : tensor<?x10xf32>
//    %0_new = disc_shape.tie_shape(%0, %0_d0, %0_d1) : tensor<?x10xf32>
//    use(%0_new)
//  ```
LogicalResult insertTieShapeOnValue(Value value, OpBuilder& b,
                                    const Location& loc) {
  // Only insert tie_shape ops for non-zero ranked tensor type
  auto ty = value.getType().dyn_cast<RankedTensorType>();
  if (!ty || ty.getRank() == 0) return success();

  DenseSet<Operation*> dimOps;
  SmallVector<Value> dimSizes;
  for (int dim = 0, rank = ty.getRank(); dim < rank; ++dim) {
    auto dimOp = b.create<tensor::DimOp>(loc, value, dim);
    dimOps.insert(dimOp);
    dimSizes.push_back(dimOp.getResult());
  }

  Value newValue = b.create<disc_shape::TieShapeOp>(loc, ty, value, dimSizes);
  auto users = llvm::to_vector<4>(value.getUsers());
  for (Operation* user : users) {
    // skip those dim ops used to fetch the dim size values of original shaped
    // value.
    if (dimOps.find(user) != dimOps.end()) continue;
    if (user == newValue.getDefiningOp()) continue;
    user->replaceUsesOfWith(value, newValue);
  }
  return success();
}

// forward declaration
LogicalResult insertTieShapeOnRegion(Region* region);

LogicalResult insertTieShapeOnOperation(Operation* op) {
  if (isa<disc_shape::TieShapeOp>(op)) return success();
  // recursively visit the regions of the op.
  if (!isa<mhlo::ReduceOp, mhlo::ReduceWindowOp>(op)) {
    for (Region& region : op->getRegions())
      if (failed(insertTieShapeOnRegion(&region)))
        return op->emitError("fail to insert tie shape for op's region\n");
  }

  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  for (Value v : op->getResults()) {
    if (failed(insertTieShapeOnValue(v, b, op->getLoc())))
      return op->emitError("fail to insert tie shape for op's result\n");
  }

  return success();
}

LogicalResult insertTieShapeOnBlock(Block* block) {
  // mapping block arguments
  OpBuilder b(block, block->begin());
  Location loc = block->getParentOp()->getLoc();
  for (Value value : block->getArguments()) {
    if (failed(insertTieShapeOnValue(value, b, loc))) {
      return block->getParentOp()->emitError(
          "failed to insert tie_shape op for block arg");
    }
  }

  // mapping each op inside the block
  // save a snapshot before visiting in case new ops are inserted during
  // visiting.
  SmallVector<Operation*> op_list;
  for (Operation& op : *block) op_list.push_back(&op);
  for (Operation* op : op_list) {
    if (failed(insertTieShapeOnOperation(op))) return failure();
  }
  return success();
}

LogicalResult insertTieShapeOnRegion(Region* region) {
  for (Block& block : *region) {
    if (failed(insertTieShapeOnBlock(&block))) return failure();
  }
  return success();
}

// convert:
//   %shape = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
// to:
//   %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//   %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//   %shape = tensor.from_elements %d0, %d1 : tensor<2xindex>
struct ExpandShapeOfOpPattern : public OpRewritePattern<shape::ShapeOfOp> {
  using OpRewritePattern<shape::ShapeOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter& rewriter) const override {
    auto ty = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!ty || !ty.hasStaticShape() || !ty.getElementType().isIndex())
      return failure();

    SmallVector<Value> dimSizes;
    for (int dim = 0, rank = ty.getShape()[0]; dim < rank; ++dim)
      dimSizes.push_back(
          rewriter.create<tensor::DimOp>(op.getLoc(), op.getArg(), dim));

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, dimSizes);
    return success();
  }
};

/// Fold dim of an operation that implements the InferShapedTypeOpInterface
template <typename OpTy>
struct DimOfShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter& rewriter) const override {
    OpResult dimValue = dimOp.getSource().template dyn_cast<OpResult>();
    if (!dimValue) return failure();
    auto shapedTypeOp =
        dyn_cast<InferShapedTypeOpInterface>(dimValue.getOwner());
    if (!shapedTypeOp) return failure();

    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex) return failure();

    SmallVector<Value> reifiedResultShapes;
    if (failed(shapedTypeOp.reifyReturnTypeShapes(
            rewriter, shapedTypeOp->getOperands(), reifiedResultShapes)))
      return failure();

    if (reifiedResultShapes.size() != shapedTypeOp->getNumResults())
      return failure();

    Value resultShape = reifiedResultShapes[dimValue.getResultNumber()];
    auto resultShapeType = resultShape.getType().dyn_cast<RankedTensorType>();
    if (!resultShapeType ||
        (!resultShapeType.getElementType().isa<IndexType>() &&
         !resultShapeType.getElementType().isa<IntegerType>()))
      return failure();

    Location loc = dimOp.getLoc();
    Value newValue = rewriter.create<tensor::ExtractOp>(
        loc, resultShape,
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, *dimIndex));
    if (!newValue.getType().isa<IndexType>())
      newValue = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), newValue);

    rewriter.replaceOp(dimOp, newValue);
    return success();
  }
};

// materialize shape computation IR by resolving tensor.dim + op having
// interface InferShapedTypeOpInterface: before
//  ```
//    %arg0 = ... : tensor<?x10xf32>
//    %0 = mhlo.transpose(%arg0) : tensor<?x?xf32>
//    %0_d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//    %0_d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//    %0_new = disc_shape.tie_shape(%0, %0_d0, %0_d1) : tensor<?x?xf32>
//    use(%0_new)
//  ```
// after materializling mhlo.transpose shape interface
//  ```
//    %arg0 = ... : tensor<?x10xf32>
//    %0 = mhlo.transpose(%arg0) {permutaion = [1, 0]} : tensor<?x?xf32>
//    %0_d0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
//    %0_d1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
//    %0_new = disc_shape.tie_shape(%0, %0_d0, %0_d1) : tensor<?x?xf32>
//    use(%0_new)
//  ```
LogicalResult materializeShapeComputation(ModuleOp m, FuncOp main) {
  // Currently we call inline before all disc passes and thus we do not need to
  // worry about function call ops. Re-visit this once we change the strategy.
  if (failed(insertTieShapeOnRegion(&main.getBody()))) {
    return failure();
  }

  RewritePatternSet patterns(m.getContext());
  // clang-format off
  patterns.add<
      ExpandShapeOfOpPattern,
      DimOfShapedTypeOpInterface<tensor::DimOp>
  >(patterns.getContext());
  // clang-format on

  if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
    return m.emitError() << "fail to materialize shape computation\n";
  }
  return success();
}

/////////////////////////// Stage #1 END //////////////////////////////////////

/////////////////////////// Stage #2 BEGIN ////////////////////////////////////

using PassPipelineRunner =
    std::function<LogicalResult(OpPassManager&, ModuleOp)>;

// Returns true if the type is possible to be a shape tensor type.
// Here shape tensor type is defined as follow:
// - rank-1 static-shaped tensor type
// - element type of the tensor is int or index
// - number of elements of the tensor < 32, supposing that the
//   higiest possible rank is smaller than 32.
bool isCandidateShapeTensorType(Type ty) {
  // Try to check if it's a candidate shape tensor.
  auto tensorTy = ty.dyn_cast<RankedTensorType>();

  return (tensorTy && tensorTy.getRank() == 1 && tensorTy.hasStaticShape() &&
          tensorTy.getElementType().isIntOrIndex() &&
          tensorTy.getShape()[0] < 32);
}

// convert:
//   %1 = disc_shape.tie_shape %0, %d0, %d1, ... : (tensor<?x?xf32>, ...) ->
//   tensor<?x?xf32> %dim_size = tensor.dim %1[%c0] : tensor<2xindex>
//   use(%dim_size)
// to:
//   use(%d0)
struct DimOfTieShapeOpCanonicalizationPattern
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter& rewriter) const override {
    auto tieShapeOp = op.getSource().getDefiningOp<disc_shape::TieShapeOp>();
    if (!tieShapeOp) return failure();
    Optional<int64_t> dimIndex = op.getConstantIndex();
    if (!dimIndex) return failure();
    rewriter.replaceOp(op, tieShapeOp->getOperand(1 + *dimIndex));
    return success();
  }
};

// convert:
//   %1 = disc_shape.tie_shape %0, %d0, %d1, ...
//         : (tensor<?x?xf32>, ...) -> tensor<?x?xf32>
//   %2 = tensor.extract %1[...] : ...
// to:
//   %2 = tensor.extract %0[...] : ...
struct ExtractElementOfTieShapeOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tieShapeOp = op.getTensor().getDefiningOp<disc_shape::TieShapeOp>();
    if (!tieShapeOp) return failure();
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, tieShapeOp->getOperand(0), op.getIndices());
    return success();
  }
};

// convert:
//   %3 = mhlo.concatenate %0, %1, %2
//         : (tensor<i32>, ...) -> tensor<3xi32>
//   %4 = tensor.extract %3[%c1] : ...
// to:
//   %4 = tensor.extract %1[%c0] : ...
struct ExtractElementOfConcatOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getIndices().size() != 1) return failure();
    auto indexOp = dyn_cast_or_null<arith::ConstantOp>(
        op.getIndices().front().getDefiningOp());
    if (!indexOp) return failure();
    int64_t index = indexOp.getValue().cast<IntegerAttr>().getInt();

    auto concatOp = op.getTensor().getDefiningOp<mhlo::ConcatenateOp>();
    if (!concatOp) return failure();
    if (!isCandidateShapeTensorType(op.getTensor().getType())) return failure();

    for (Value operand : concatOp->getOperands()) {
      if (!isCandidateShapeTensorType(operand.getType())) return failure();
      auto operandTy = operand.getType().cast<RankedTensorType>();
      if (index >= operandTy.getNumElements()) {
        index -= operandTy.getNumElements();
        continue;
      }

      Value newIndex =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), index);
      Value newValue =
          rewriter.create<tensor::ExtractOp>(op.getLoc(), operand, newIndex);
      rewriter.replaceOp(op, {newValue});
      return success();
    }
    return failure();
  }
};

// convert:
//   %1 = mhlo.slice(%0) {start_indices = dense<1>, ...}
//         : (tensor<3xi32>) -> tensor<2xi32>
//   %2 = tensor.extract %1[%c1] : ...
// to:
//   %2 = tensor.extract %0[%c2] : ...
struct ExtractElementOfSliceOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getIndices().size() != 1) return failure();
    auto indexOp = dyn_cast_or_null<arith::ConstantOp>(
        op.getIndices().front().getDefiningOp());
    if (!indexOp) return failure();
    int64_t index = indexOp.getValue().cast<IntegerAttr>().getInt();

    auto sliceOp = op.getTensor().getDefiningOp<mhlo::SliceOp>();
    if (!sliceOp) return failure();
    if (!isCandidateShapeTensorType(op.getTensor().getType())) return failure();

    int64_t start = sliceOp.getStartIndices().getValues<int64_t>()[0];
    int64_t stride = sliceOp.getStrides().getValues<int64_t>()[0];
    int64_t inputIndex = (start + index * stride);
    Value newIndex =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), inputIndex);
    Value newValue = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), sliceOp.getOperand(), newIndex);

    rewriter.replaceOp(op, {newValue});
    return success();
  }
};

// convert:
//   %1 = mhlo.reshape(%0) : (tensor<i32>) -> tensor<1xi32>
//   %2 = tensor.extract %1[%c0] : ...
// to:
//   %2 = tensor.extract %0[] : ...
struct ExtractElementOfReshapeOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tensorTy = op.getTensor().getType().dyn_cast<RankedTensorType>();
    if (!tensorTy || !tensorTy.hasStaticShape() ||
        !tensorTy.getElementType().isIntOrIndex() ||
        tensorTy.getNumElements() > 8 || tensorTy.getNumElements() == 0)
      return failure();

    auto reshapeOp = op.getTensor().getDefiningOp<mhlo::ReshapeOp>();
    if (!reshapeOp) return failure();

    SmallVector<int64_t> indices;
    for (Value indexValue : op.getIndices()) {
      auto indexOp = indexValue.getDefiningOp<arith::ConstantOp>();
      if (!indexOp) return failure();
      indices.push_back(indexOp.getValue().cast<IntegerAttr>().getInt());
    }

    int linearIndex = indices.empty() ? 0 : indices[0];
    for (size_t i = 1; i < indices.size(); ++i) {
      linearIndex = linearIndex * tensorTy.getShape()[i] + indices[i];
    }

    SmallVector<int64_t> inputIndices;
    auto inputTy =
        reshapeOp.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!inputTy || !inputTy.hasStaticShape()) return failure();
    for (int i = inputTy.getRank() - 1; i >= 0; --i) {
      inputIndices.push_back(linearIndex % inputTy.getShape()[i]);
      linearIndex /= inputTy.getShape()[i];
    }
    SmallVector<Value> newIndices;
    for (int i = inputTy.getRank() - 1; i >= 0; --i) {
      newIndices.push_back(rewriter.create<arith::ConstantIndexOp>(
          op.getLoc(), inputIndices[i]));
    }
    Value newValue = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), reshapeOp.getOperand(), newIndices);
    rewriter.replaceOp(op, {newValue});
    return success();
  }
};

// convert:
//   %1 = mhlo.reduce(%0) applies mhlo.multiply across dimensions = [0]
//        : (tensor<2x1xi32>, tensor<i32>) -> tensor<1xi32>
//   %2 = tensor.extract %1[%c0] : ...
//   use(%2)
// to:
//   %1 = tensor.extract %0[%c0, %c0] : ...
//   %2 = tensor.extract %0[%c1, %c0] : ...
//   %3 = arith.muli %1, %2
//   use(%3)
//
// This pattern is usually genearted when lowering a TF pattern like:
//   %1 = tf.Shape(%0) : tensor<3xi32>
//   %2 = tf.Slice(%1, ...) : tensor<2xi32>
//   %3 = tf.Reshape(%2) : tensor<2x1xi32>
//   %4 = tf.Prod(%3) : tensor<1xi32>
//   use(%4)
struct ExtractElementOfReduceOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tensorTy = op.getTensor().getType().dyn_cast<RankedTensorType>();
    if (!tensorTy || !tensorTy.hasStaticShape() ||
        !tensorTy.getElementType().isIntOrIndex() ||
        tensorTy.getNumElements() > 8 || tensorTy.getNumElements() == 0)
      return failure();

    auto reduceOp = op.getTensor().getDefiningOp<mhlo::ReduceOp>();
    if (!reduceOp || reduceOp->getNumResults() > 1) return failure();

    // Only support reducing arcross a single dimension.
    if (reduceOp.getDimensions().getValues<int64_t>().size() != 1)
      return failure();
    int64_t reduceAxis = *reduceOp.getDimensions().getValues<int64_t>().begin();

    auto& block = reduceOp.getBody().front();
    if (!hasSingleElement(block.without_terminator())) return failure();
    if (!isa<mhlo::MulOp>(&(*block.begin()))) return failure();

    Value dataTensor = reduceOp->getOperand(0);
    Value initTensor = reduceOp->getOperand(1);
    auto initTy = initTensor.getType().dyn_cast<RankedTensorType>();
    auto dataTy = dataTensor.getType().dyn_cast<RankedTensorType>();
    if (!initTy || initTy.getRank() > 1 || !dataTy ||
        !dataTy.hasStaticShape() || dataTy.getNumElements() > 8)
      return failure();

    Value initValue;
    Location loc = op.getLoc();
    if (initTy.getRank() == 0) {
      initValue = rewriter.create<tensor::ExtractOp>(loc, initTensor);
    } else {
      assert(initTy.getRank() == 1);
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      initValue = rewriter.create<tensor::ExtractOp>(loc, initTensor, zero);
    }

    int nextNonReduceIdx = 0;
    SmallVector<Value> indices(dataTy.getRank());
    for (int64_t i = 0; i < dataTy.getRank(); ++i) {
      if (i == reduceAxis) continue;
      indices[i] = op.getIndices()[nextNonReduceIdx++];
    }

    Value newResult = initValue;
    for (int64_t i = 0; i < dataTy.getShape()[reduceAxis]; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      indices[reduceAxis] = idx;
      Value data = rewriter.create<tensor::ExtractOp>(loc, dataTensor, indices);
      newResult = rewriter.create<arith::MulIOp>(loc, newResult, data);
    }
    rewriter.replaceOp(op, {newResult});
    return success();
  }
};

// convert:
//   // functionally like a slice op
//   %1 = mhlo.gather(%0, ...) : (tensor<3xi32>) -> tensor<2xi32>
//   %2 = tensor.extract %1[%c0] : ...
//   use(%2)
// to:
//   %1 = tensor.extract %0[%c1] : ...
//   use(%1)
//
// This pattern is usually genearted when lowering a TF pattern like:
//   %1 = tf.Shape(%0) : tensor<3xi32>
//   %2 = tf.Gather(%1, ...) : tensor<2xi32> // functionally like a slice op
//   %3 = tf.Reshape(%2) : tensor<2x1xi32>
//   %4 = tf.Prod(%3) : tensor<1xi32>
//   use(%4)
struct ExtractElementOfGatherOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getIndices().size() != 1) return failure();
    auto indexOp = dyn_cast_or_null<arith::ConstantOp>(
        op.getIndices().front().getDefiningOp());
    if (!indexOp) return failure();
    int64_t index = indexOp.getValue().cast<IntegerAttr>().getInt();

    auto gatherOp = op.getTensor().getDefiningOp();
    if (!gatherOp || !isa<mhlo::GatherOp, mhlo::DynamicGatherOp>(gatherOp))
      return failure();

    Value in = gatherOp->getOperand(0);
    Value startIndices = gatherOp->getOperand(1);
    Value out = gatherOp->getResult(0);
    if (!isCandidateShapeTensorType(in.getType()) ||
        !isCandidateShapeTensorType(startIndices.getType()) ||
        !isCandidateShapeTensorType(out.getType()))
      return failure();

    auto dimensionNumbers =
        gatherOp->getAttrOfType<mhlo::GatherDimensionNumbersAttr>(
            "dimension_numbers");
    auto collapsedSliceDims = dimensionNumbers.getCollapsedSliceDims();
    auto indexVectorDim = dimensionNumbers.getIndexVectorDim();
    auto startIndexMap = dimensionNumbers.getStartIndexMap();
    auto offsetDims = dimensionNumbers.getOffsetDims();

    // TODO(disc): support other cases.
    if (collapsedSliceDims.size() != 1 || collapsedSliceDims[0] != 0 ||
        indexVectorDim != 1 || startIndexMap.size() != 1 ||
        startIndexMap[0] != 0 || offsetDims.size() != 0) {
      return failure();
    }

    Location loc = op.getLoc();
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, index);
    Value newIndex =
        rewriter.create<tensor::ExtractOp>(loc, startIndices, offset);
    if (newIndex.getType() != rewriter.getIndexType())
      newIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), newIndex);
    Value newValue = rewriter.create<tensor::ExtractOp>(loc, in, newIndex);
    rewriter.replaceOp(op, {newValue});

    return success();
  }
};

// convert:
//   %1 = arith.index_cast(%0) : (tensor<2xi32>) -> tensor<2xindex>
//   use(%1)
// to:
//   %0 = tensor.extract %0[%c0] : tensor<2xi32>
//   %1 = arith.index_cast %0 : index
//   %2 = tensor.extract %0[%c1] : tensor<2xi32>
//   %3 = arith.index_cast %2 : index
//   %4 = tensor.from_elements %1, %3 : tensor<2xindex>
//   use(%4)
struct ScalarizeIndexCastOpCanonicalizationPattern
    : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter& rewriter) const override {
    if (!isCandidateShapeTensorType(op.getType())) return failure();
    Value input = op->getOperand(0);
    auto inTy = input.getType().cast<RankedTensorType>();
    auto outTy = op.getType().cast<RankedTensorType>();

    SmallVector<Value> elems;
    for (int i = 0; i < inTy.getNumElements(); ++i) {
      Value index = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      Value inElem =
          rewriter.create<tensor::ExtractOp>(op.getLoc(), input, index);
      elems.push_back(rewriter.create<arith::IndexCastOp>(
          op.getLoc(), outTy.getElementType(), inElem));
    }

    Value newOut =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), outTy, elems);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, outTy, elems);
    return success();
  }
};

// Adds shape optimization related patterns.
void populateShapeOptimizationPatterns(MLIRContext* context,
                                       RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      DimOfTieShapeOpCanonicalizationPattern,
      ExtractElementOfConcatOpCanonicalizationPattern,
      ExtractElementOfGatherOpCanonicalizationPattern,
      ExtractElementOfReduceOpCanonicalizationPattern,
      ExtractElementOfReshapeOpCanonicalizationPattern,
      ExtractElementOfSliceOpCanonicalizationPattern,
      ExtractElementOfTieShapeOpCanonicalizationPattern,
      ScalarizeIndexCastOpCanonicalizationPattern
  >(patterns->getContext());
  // clang-format on
}

// Adds canonicalization patterns to the list of patterns.
void addCanonicalizationPatterns(MLIRContext* context,
                                 RewritePatternSet* patterns) {
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

LogicalResult runCanonicalizer(ModuleOp m, PassPipelineRunner runner) {
  MLIRContext* context = m.getContext();
  RewritePatternSet patterns(context);
  populateShapeOptimizationPatterns(context, &patterns);
  addCanonicalizationPatterns(context, &patterns);

  SmallVector<std::string> disablePatterns = {
      "mlir::disc_shape::{anonymous}::IdentityTieShapeOp"};
  FrozenRewritePatternSet frozenSet(std::move(patterns), disablePatterns);

  if (failed(applyPatternsAndFoldGreedily(m, std::move(frozenSet)))) {
    return m.emitError() << "fail to run canonicalizer\n";
  }

  OpPassManager dynamicPM("builtin.func");
  dynamicPM.addPass(createCSEPass());
  return runner(dynamicPM, m);
}

class ShapeComputationIRAnalysis {
 public:
  explicit ShapeComputationIRAnalysis(FuncOp func, SymbolicDimMgr& mgr);

  // Analyzes the shape computation IR and the shape constraint IR in the target
  // module Returns failure if failed.
  LogicalResult run();

  // Returns the func op this analysis object runs on.
  FuncOp getFunc() { return funcOp_; }

  // Each dim size value (as one shape operand of one disc_shape.tie_shape) is
  // mapped to one SymbolicDim after the analysis. This function group all such
  // dim size values by the SymbolicDim, and return the results.
  DenseMap<SymbolicDimOp, SmallVector<Value>> getSymbolicDimSSAValueInstance();

  Type getRefinedType(Value value);

  SymbolicDimOp value2SymbolicDimOp(Value value);

  llvm::Optional<SmallVector<SymbolicDimOp>> rankedTensor2SymDims(Value value);

 private:
  LogicalResult runOnRegion(Region* region);
  LogicalResult runOnBlock(Block* block);
  LogicalResult runOnOperation(Operation* op);

  LogicalResult buildSymbolicShape(Value value);
  LogicalResult buildSymbolicShapeForResultsOfOp(Operation* op);

  LogicalResult applyOpConstraint(Operation* op);
  LogicalResult applyIndexOpConstraint(Operation* op);
  LogicalResult applyShapeTensorOpConstraint(Operation* op);
  LogicalResult applyRankedTensorOpConstraint(Operation* op);
  LogicalResult applyMhloOpConstraint(Operation* op);
  LogicalResult applyMhloElemOpConstraint(Operation* op);
  LogicalResult applyMhloDotLikeOpConstraint(Operation* op);
  LogicalResult applyMhloBcastOpConstraint(Operation* op);
  LogicalResult applyMhloConcatOpConstraint(Operation* op);
  LogicalResult applyMhloReshapeLikeOpConstraint(Operation* op);

  LogicalResult applyTieShapeOpConstraint(Operation* op);
  LogicalResult applyTieShapeOfReshapePatternConstraint(Operation* op);
  LogicalResult tryToBuildProductEqualityForReshape(
      Operation* reshapeOp, const SmallVectorImpl<SymbolicDimExpr>& inExprs,
      int inStartIdx, const SmallVectorImpl<SymbolicDimExpr>& outExprs,
      int outStartIdx);

  LogicalResult mapRankedValueShapeEqual(Value lhs, Value rhs);

 private:
  bool initialized_ = false;
  FuncOp funcOp_;
  SymbolicDimMgr& mgr_;

  // Map scalar int/index SSA value to a symbolicDim
  DenseMap<Value, SymbolicDimOp> value2SymDim_;

  // Map a shape tensor value (1D ranked int/index tensor) to an array of
  // symbolicDims, each for one component of the shape tensor.
  DenseMap<Value, SmallVector<SymbolicDimOp>> shapeTensor2SymDims_;

  // Map a ranked tensor value to an array of symbolicDims, each represents one
  // dimension size of the tensor.
  DenseMap<Value, SmallVector<SymbolicDimOp>> rankedTensor2SymDims_;

  // Map an index value to its defining symbolic expr;
  DenseMap<Value, SymbolicDimExpr> value2DefiningExpr_;
};

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(FuncOp func,
                                                       SymbolicDimMgr& mgr)
    : funcOp_(func), mgr_(mgr) {}

LogicalResult ShapeComputationIRAnalysis::run() {
  // Make sure only run once.
  if (initialized_) {
    return funcOp_->emitError()
           << "re-initialized shape analysis is not supported\n";
  }
  initialized_ = true;
  return runOnRegion(&funcOp_.getBody());
}

LogicalResult ShapeComputationIRAnalysis::runOnRegion(Region* region) {
  // Only SCF is supported a.t.m.
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitError(
        "only single block region is supported");
  }
  for (Block& block : *region) {
    if (failed(runOnBlock(&block))) return failure();
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::runOnBlock(Block* block) {
  // mapping block arguments
  for (Value value : block->getArguments()) {
    if (failed(buildSymbolicShape(value))) {
      return block->getParentOp()->emitError(
          "failed to build shape for block arg");
    }
  }

  // mapping each op inside the block
  // save a snapshot before visiting in case new ops are inserted during
  // visiting.
  SmallVector<Operation*> op_list;
  for (Operation& op : *block) op_list.push_back(&op);
  for (Operation* op : op_list) {
    if (failed(runOnOperation(op))) return failure();
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::runOnOperation(Operation* op) {
  LLVM_DEBUG(llvm::dbgs() << "runOnOperation: " << *op << "\n");
  if (failed(buildSymbolicShapeForResultsOfOp(op)))
    return op->emitError() << "fail to buildSymbolicShapeForResultsOfOp\n";

  // TODO(disc): visit the regions of op once we support funcitonal control
  // flow.

  // apply op's shape constraint
  return applyOpConstraint(op);
}

LogicalResult ShapeComputationIRAnalysis::buildSymbolicShapeForResultsOfOp(
    Operation* op) {
  // build shapes for the results of op
  for (Value result : op->getResults()) {
    if (failed(buildSymbolicShape(result))) {
      return op->emitError("failed to build shape for op's result");
    }
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::buildSymbolicShape(Value value) {
  Type ty = value.getType();
  if (ty.isIntOrIndex()) {
    SymbolicDimOp sym = mgr_.newSymbolicDim();
    value2SymDim_[value] = sym;
    value2DefiningExpr_[value] = SymbolicDimExpr(value);
  } else if (auto tensorTy = ty.dyn_cast<RankedTensorType>()) {
    SmallVector<SymbolicDimOp> symbols =
        mgr_.getOrCreateSymbolicDimsForRankedValue(value);
    rankedTensor2SymDims_[value] = std::move(symbols);
    // Try to check if it's a candidate shape tensor.
    if (isCandidateShapeTensorType(ty)) {
      SmallVector<SymbolicDimOp> symbols;
      for (int i = 0, d = tensorTy.getShape()[0]; i < d; ++i)
        symbols.push_back(mgr_.newSymbolicDim());
      shapeTensor2SymDims_[value] = std::move(symbols);
    }
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::mapRankedValueShapeEqual(Value lhs,
                                                                   Value rhs) {
  auto& lhsDims = rankedTensor2SymDims_[lhs];
  auto& rhsDims = rankedTensor2SymDims_[rhs];

  if (lhsDims.size() != rhsDims.size()) {
    return mlir::emitError(lhs.getLoc())
           << "miss match rank:\n\t"
           << "lhs = " << lhs << "\n\trhs = " << rhs << "\n";
  }

  for (const auto& en : llvm::zip(lhsDims, rhsDims)) {
    if (failed(mgr_.mapSymbolicDimEqual(std::get<0>(en), std::get<1>(en))))
      return mlir::emitError(lhs.getLoc())
             << "fail to merge:\n\t"
             << "lhs = " << lhs << "\n\trhs = " << rhs << "\n";
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyIndexOpConstraint(
    Operation* op) {
  if (op->getNumResults() == 0) return success();

  // Only handle scalar int/index op
  Type ty = op->getResult(0).getType();
  if (!ty.isIntOrIndex()) return success();

  if (isa<arith::IndexCastOp, arith::TruncIOp>(op)) {
    Value in = op->getOperand(0);
    Value out = op->getResult(0);
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[in], value2SymDim_[out])))
      return op->emitError() << "fail to merge dim\n";
    value2DefiningExpr_[out] = value2DefiningExpr_[in];
  } else if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex) return success();
    value2SymDim_[op->getResult(0)].updateKnownNonNegative(true);
    if (failed(mgr_.mapSymbolicDimEqual(
            value2SymDim_[op->getResult(0)],
            rankedTensor2SymDims_[dimOp.getSource()][*dimIndex])))
      return op->emitError() << "fail to merge dim\n";
  } else if (isa<arith::ConstantIndexOp, arith::ConstantIntOp>(op)) {
    int64_t val = op->getAttrOfType<IntegerAttr>("value").getInt();
    LLVM_DEBUG(llvm::dbgs() << "applyIndexOpConstraint arith const op val = "
                            << val << "\n");
    Value out = op->getResult(0);
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[out],
                                        mgr_.newConstantSymbolicDim(val))))
      return op->emitError() << "fail to merge dim\n";
    value2DefiningExpr_[out] = SymbolicDimExpr(val, out.getContext());
  } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (!isCandidateShapeTensorType(extractOp.getTensor().getType()))
      return success();
    auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(
        op->getOperand(1).getDefiningOp());
    if (!indexOp) return success();
    int64_t index = indexOp.getValue().cast<IntegerAttr>().getInt();
    auto& shapeTensorDims = shapeTensor2SymDims_[extractOp.getTensor()];
    if (index >= shapeTensorDims.size())
      return op->emitError() << "miss match shape tensor size\n";
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[op->getResult(0)],
                                        shapeTensorDims[index])))
      return op->emitError() << "fail to merge dim\n";
  } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value out = op->getResult(0);
    value2DefiningExpr_[out] = SymbolicDimExpr::buildMulExpr(
        value2DefiningExpr_[lhs], value2DefiningExpr_[rhs]);
    auto lhsSym = mgr_.getRootSymbolicDim(value2SymDim_[lhs]);
    auto rhsSym = mgr_.getRootSymbolicDim(value2SymDim_[rhs]);
    auto outSym = mgr_.getRootSymbolicDim(value2SymDim_[out]);
    if (lhsSym.getKnownNonNegative() && rhsSym.getKnownNonNegative()) {
      outSym.updateKnownNonNegative(true);
    }
    if (lhsSym.getKnownNonNegative() && rhsSym.getKnownNonSizeOne() ||
        rhsSym.getKnownNonNegative() && lhsSym.getKnownNonSizeOne()) {
      outSym.updateKnownNonSizeOne(true);
    }
    if (lhsSym.getKnownNonSizeZero() && rhsSym.getKnownNonSizeZero()) {
      outSym.updateKnownNonSizeZero(true);
    }
    // TODO(disc): propagate other attributes/shape ranges??
  } else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    // TODO(disc): build symbolic expression for AddIOp
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value out = op->getResult(0);
    auto lhsSym = mgr_.getRootSymbolicDim(value2SymDim_[lhs]);
    auto rhsSym = mgr_.getRootSymbolicDim(value2SymDim_[rhs]);
    auto outSym = mgr_.getRootSymbolicDim(value2SymDim_[out]);
    if (lhsSym.getKnownNonNegative() && rhsSym.getKnownNonNegative()) {
      outSym.updateKnownNonNegative(true);
      if (lhsSym.getKnownNonSizeZero() || rhsSym.getKnownNonSizeZero())
        outSym.updateKnownNonSizeZero(true);
    }
    // TODO(disc): propagate other constraint attributes/shape ranges??
  }

  // TODO: add support for arith::subi/divi/select...

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyShapeTensorOpConstraint(
    Operation* op) {
  if (isa<tensor::FromElementsOp>(op)) {
    // FromElementsOp supports tensor with any rank. We only want to process
    // those `FromElementsOp` which are likely to be shape tensor.
    if (!isCandidateShapeTensorType(op->getResult(0).getType())) {
      return success();
    }
    auto& symbols = shapeTensor2SymDims_[op->getResult(0)];
    if (symbols.size() != op->getOperands().size())
      return op->emitError()
             << "miss match dim size and num operands: " << symbols.size()
             << " vs " << op->getNumOperands();
    for (const auto& z : llvm::zip(op->getOperands(), symbols)) {
      if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[std::get<0>(z)],
                                          std::get<1>(z))))
        return op->emitError() << "fail to merge dim\n";
    }
  } else if (isa<shape::BroadcastOp>(op)) {
    // TODO(disc): support more than two operands
    if (op->getNumOperands() != 2) return success();
    auto& lhsDims = shapeTensor2SymDims_[op->getOperand(0)];
    auto& rhsDims = shapeTensor2SymDims_[op->getOperand(1)];
    auto& outDims = shapeTensor2SymDims_[op->getResult(0)];

    SymbolicDimOp sizeOneDim = mgr_.newConstantSymbolicDim(1);
    int lhsRank = static_cast<int>(lhsDims.size());
    int rhsRank = static_cast<int>(rhsDims.size());
    int outRank = static_cast<int>(outDims.size());
    for (int d = outRank - 1; d >= 0; --d) {
      int reverseD = outRank - 1 - d;
      SymbolicDimOp lhsSymbol =
          ((reverseD < lhsRank) ? lhsDims[lhsRank - 1 - reverseD] : sizeOneDim);
      SymbolicDimOp rhsSymbol =
          ((reverseD < rhsRank) ? rhsDims[rhsRank - 1 - reverseD] : sizeOneDim);
      SymbolicDimOp outSymbol = outDims[d];

      auto getRoot = [&](SymbolicDimOp sym) {
        return mgr_.getRootSymbolicDim(sym);
      };

      if (getRoot(lhsSymbol) == getRoot(rhsSymbol))
        if (failed(mgr_.mapSymbolicDimEqual(lhsSymbol, outSymbol)))
          return op->emitError() << "fail to merge dim\n";
      if (getRoot(lhsSymbol) == getRoot(sizeOneDim))
        if (failed(mgr_.mapSymbolicDimEqual(rhsSymbol, outSymbol)))
          return op->emitError() << "fail to merge dim\n";
      if (getRoot(rhsSymbol) == getRoot(sizeOneDim))
        if (failed(mgr_.mapSymbolicDimEqual(lhsSymbol, outSymbol)))
          return op->emitError() << "fail to merge dim\n";
    }
  } else if (auto computeReshapeOp =
                 dyn_cast<mhlo::ComputeReshapeShapeOp>(op)) {
    auto& inDims = shapeTensor2SymDims_[op->getOperand(1)];
    auto& outDims = shapeTensor2SymDims_[op->getResult(0)];

    if (inDims.size() != outDims.size()) {
      return op->emitError() << "ComputeReshapeShapeOp mismatch rank\n";
    }

    for (auto [inSymbol, outSymbol] : llvm::zip(inDims, outDims)) {
      if (mgr_.getRootSymbolicDim(inSymbol).getKnownNonNegative()) {
        if (failed(mgr_.mapSymbolicDimEqual(inSymbol, outSymbol)))
          return op->emitError() << "fail to merge dim\n";
      }
    }
  }

  // TODO: add support for arith::addi/subi/...

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloElemOpConstraint(
    Operation* op) {
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() ||
      op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
    Value ref;
    if (op->getNumOperands() > 0) ref = op->getOperands().front();
    if (op->getNumResults() > 0) ref = op->getResults().front();
    if (!ref) return success();
    for (Value operand : op->getOperands())
      if (failed(mapRankedValueShapeEqual(operand, ref)))
        return op->emitError()
               << "fail to merge symbolic dim of operand of element op\n";
    for (Value result : op->getResults())
      if (failed(mapRankedValueShapeEqual(result, ref)))
        return op->emitError()
               << "fail to merge symbolic dim of result of element op\n";
  } else if (op->hasTrait<mlir::OpTrait::SameTypeOperands>() ||
             op->hasTrait<mlir::OpTrait::SameOperandsShape>()) {
    if (op->getNumOperands() == 0) return success();
    Value ref = op->getOperands().front();
    for (Value operand : op->getOperands())
      if (failed(mapRankedValueShapeEqual(operand, ref)))
        return op->emitError()
               << "fail to merge symbolic dim between operands of element op\n";
  } else if (auto clamp = dyn_cast<mhlo::ClampOp>(op)) {
    auto operandTy = clamp.getOperand().getType().dyn_cast<RankedTensorType>();
    auto minTy = clamp.getMin().getType().dyn_cast<RankedTensorType>();
    auto maxTy = clamp.getMax().getType().dyn_cast<RankedTensorType>();
    if (!operandTy || !minTy || !maxTy) return success();

    if (minTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(clamp.getOperand(), clamp.getMin())))
        return op->emitError()
               << "fail to merge the symbolic dim of operand and "
                  "min of mhlo::ClampOp\n";
    }
    if (maxTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(clamp.getOperand(), clamp.getMax())))
        return op->emitError()
               << "fail to merge the symbolic dim of operand and "
                  "max of mhlo::ClampOp\n";
    }
  } else if (auto select = dyn_cast<mhlo::SelectOp>(op)) {
    auto predTy = select.getPred().getType().dyn_cast<RankedTensorType>();
    auto trueTy = select.getOnTrue().getType().dyn_cast<RankedTensorType>();
    auto falseTy = select.getOnFalse().getType().dyn_cast<RankedTensorType>();
    auto resultTy = select.getResult().getType().dyn_cast<RankedTensorType>();
    if (!predTy || !trueTy || !falseTy || !resultTy) return success();

    if (predTy.getRank() != 0) {
      if (failed(
              mapRankedValueShapeEqual(select.getPred(), select.getResult())))
        return op->emitError() << "fail to merge the symbolic dim of pred and "
                                  "result of mhlo::SelectOp\n";
    }
    if (trueTy.getRank() != 0) {
      if (failed(
              mapRankedValueShapeEqual(select.getOnTrue(), select.getResult())))
        return op->emitError()
               << "fail to merge the symbolic dim of on_true and "
                  "result of mhlo::SelectOp\n";
    }
    if (falseTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(select.getOnFalse(),
                                          select.getResult())))
        return op->emitError()
               << "fail to merge the symbolic dim of on_false and "
                  "result of mhlo::SelectOp\n";
    }
  }

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloDotLikeOpConstraint(
    Operation* op) {
  if (auto dot_general = dyn_cast<mhlo::DotGeneralOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return success();
    auto& lhsDims = rankedTensor2SymDims_[lhs];
    auto& rhsDims = rankedTensor2SymDims_[rhs];
    if (lhsTy.getRank() != lhsDims.size() || rhsTy.getRank() != rhsDims.size())
      return op->emitError("lhs or rhs mismatch rank\n");
    auto dim_numbers = dot_general.getDotDimensionNumbers();
    // Contracting dimensions.
    auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    assert(lhs_contracting_dims.size() == rhs_contracting_dims.size());
    for (int64_t i = 0; i < lhs_contracting_dims.size(); i++) {
      int64_t lhs_dim = lhs_contracting_dims[i];
      int64_t rhs_dim = rhs_contracting_dims[i];
      if (failed(mgr_.mapSymbolicDimEqual(lhsDims[lhs_dim], rhsDims[rhs_dim])))
        return op->emitError() << "fail to merge dim\n";
    }
    // Batching dimensions.
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    assert(lhs_batching_dims.size() == rhs_batching_dims.size());
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      int64_t rhs_dim = rhs_batching_dims[i];
      if (failed(mgr_.mapSymbolicDimEqual(lhsDims[lhs_dim], rhsDims[rhs_dim])))
        return op->emitError() << "fail to merge dim\n";
    }
  } else if (auto dot = dyn_cast<mhlo::DotOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return success();
    auto& lhsDims = rankedTensor2SymDims_[lhs];
    auto& rhsDims = rankedTensor2SymDims_[rhs];
    if (lhsTy.getRank() != lhsDims.size() || rhsTy.getRank() != rhsDims.size())
      return op->emitError("lhs or rhs mismatch rank\n");

    if (failed(
            mgr_.mapSymbolicDimEqual(lhsDims[lhsTy.getRank() - 1], rhsDims[0])))
      return op->emitError() << "fail to merge dim\n";
  } else if (auto einsum = dyn_cast<mhlo::EinsumOp>(op)) {
    auto lhsTy = einsum.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsTy = einsum.getRhs().getType().dyn_cast<RankedTensorType>();
    auto outTy = einsum.getResult().getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy || !outTy) return success();

    auto& lhsDims = rankedTensor2SymDims_[einsum.getLhs()];
    auto& rhsDims = rankedTensor2SymDims_[einsum.getRhs()];
    auto& outDims = rankedTensor2SymDims_[einsum.getResult()];

    StringRef equation = einsum.getEinsumConfig();
    llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>
        all_tokens;
    if (!parseEinsumEquation(equation, all_tokens, nullptr, nullptr, nullptr)) {
      return einsum.emitError("unexpected character in einsum equation");
    }
    for (auto token : all_tokens) {
      SmallVector<int64_t> equalValues(3, -1);
      for (auto item : token.second) {
        if (item.first == kIsLhs) {
          equalValues[0] = item.second;
        } else if (item.first == kIsRhs) {
          equalValues[1] = item.second;
        } else {
          // kIsResult
          equalValues[2] = item.second;
        }
      }
      if (equalValues[0] >= 0 && equalValues[1] >= 0) {
        int64_t lhsIdx = equalValues[0];
        int64_t rhsIdx = equalValues[1];
        if (lhsIdx >= lhsDims.size() || rhsIdx >= rhsDims.size()) {
          return op->emitError("lhs or rhs mismatch rank\n");
        }
        if (failed(mgr_.mapSymbolicDimEqual(lhsDims[lhsIdx], rhsDims[rhsIdx])))
          return op->emitError() << "fail to merge dim\n";
      }
      if (equalValues[0] >= 0 && equalValues[2] >= 0) {
        int64_t lhsIdx = equalValues[0];
        int64_t outIdx = equalValues[2];
        if (lhsIdx >= lhsDims.size() || outIdx >= outDims.size()) {
          return op->emitError("lhs or rhs mismatch rank\n");
        }
        if (failed(mgr_.mapSymbolicDimEqual(lhsDims[lhsIdx], outDims[outIdx])))
          return op->emitError() << "fail to merge dim\n";
      }
      if (equalValues[1] >= 0 && equalValues[2] >= 0) {
        int64_t rhsIdx = equalValues[1];
        int64_t outIdx = equalValues[2];
        if (rhsIdx >= rhsDims.size() || outIdx >= outDims.size()) {
          return op->emitError("lhs or rhs mismatch rank\n");
        }
        if (failed(mgr_.mapSymbolicDimEqual(rhsDims[rhsIdx], outDims[outIdx])))
          return op->emitError() << "fail to merge dim\n";
      }
    }
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloBcastOpConstraint(
    Operation* op) {
  if (isa<mhlo::DynamicBroadcastInDimOp>(op)) {
    auto& shapeTensorDims = shapeTensor2SymDims_[op->getOperand(1)];
    auto& outDims = rankedTensor2SymDims_[op->getResult(0)];

    if (shapeTensorDims.size() != outDims.size())
      return op->emitError() << "mismatch out rank and shape tensor size\n";
    for (const auto& z : llvm::zip(shapeTensorDims, outDims)) {
      if (failed(mgr_.mapSymbolicDimEqual(std::get<0>(z), std::get<1>(z))))
        return op->emitError() << "fail to merge dim\n";
    }
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloConcatOpConstraint(
    Operation* op) {
  if (auto concat = dyn_cast<mhlo::ConcatenateOp>(op)) {
    Value out = op->getResult(0);
    int64_t axis = concat.getDimension();
    auto ty = out.getType().dyn_cast<RankedTensorType>();
    if (!ty) return success();
    auto& outDims = rankedTensor2SymDims_[out];
    if (ty.getRank() != outDims.size())
      return op->emitError() << "output rank mismatch\n";
    for (Value operand : op->getOperands()) {
      auto& inDims = rankedTensor2SymDims_[operand];
      if (inDims.size() != outDims.size())
        return op->emitError() << "input and output rank mismatch\n";
      for (int64_t i = 0; i < ty.getRank(); ++i) {
        if (i == axis) continue;
        if (failed(mgr_.mapSymbolicDimEqual(inDims[i], outDims[i])))
          return op->emitError() << "fail to merge dim\n";
      }
    }
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloReshapeLikeOpConstraint(
    Operation* op) {
  if (auto dynReshape = dyn_cast<mhlo::DynamicReshapeOp>(op)) {
    Value in = op->getOperand(0);
    Value targetShape = op->getOperand(1);
    Value out = op->getResult(0);
    auto inTy = in.getType().dyn_cast<RankedTensorType>();
    auto outTy = out.getType().dyn_cast<RankedTensorType>();
    if (!inTy || !outTy) return success();

    auto& shapeTensorDims = shapeTensor2SymDims_[targetShape];
    auto& outDims = rankedTensor2SymDims_[out];
    if (shapeTensorDims.size() != outDims.size())
      return op->emitError() << "mismatch out rank and shape tensor size\n";
    for (const auto& z : llvm::zip(shapeTensorDims, outDims)) {
      if (failed(mgr_.mapSymbolicDimEqual(std::get<0>(z), std::get<1>(z))))
        return op->emitError() << "fail to merge dim\n";
    }
    if (failed(mgr_.mapSymbolicDimProductEqual(
            SymbolicDimProduct{outDims},
            SymbolicDimProduct{rankedTensor2SymDims_[in]})))
      return op->emitError() << "fail to map product equal between the operand "
                                "and result of reshape op\n";
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloOpConstraint(Operation* op) {
  if (failed(applyMhloElemOpConstraint(op))) return failure();

  if (failed(applyMhloDotLikeOpConstraint(op))) return failure();

  if (failed(applyMhloBcastOpConstraint(op))) return failure();

  if (failed(applyMhloConcatOpConstraint(op))) return failure();

  if (failed(applyMhloReshapeLikeOpConstraint(op))) return failure();

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyRankedTensorOpConstraint(
    Operation* op) {
  if (isa<tensor::CastOp>(op)) {
    auto inTy = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
    auto outTy = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!inTy || !outTy) return success();
    if (failed(mapRankedValueShapeEqual(op->getOperand(0), op->getResult(0))))
      return op->emitError() << "fail to merge the symbolic dim of operand and "
                                "result of tensor::CastOp\n";
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::tryToBuildProductEqualityForReshape(
    Operation* reshapeOp, const SmallVectorImpl<SymbolicDimExpr>& inExprs,
    int inStartIdx, const SmallVectorImpl<SymbolicDimExpr>& outExprs,
    int outStartIdx) {
  SmallVector<SymbolicDimExpr> subInExprs, subOutExprs;
  SmallVector<SymbolicDimOp> subInDims, subOutDims;
  auto& inDims = rankedTensor2SymDims_[reshapeOp->getOperand(0)];
  auto& outDims = rankedTensor2SymDims_[reshapeOp->getResult(0)];
  auto MergeExprs = [&](const SmallVector<SymbolicDimExpr>& exprs) {
    SymbolicDimExpr result = SymbolicDimExpr(1, reshapeOp->getContext());
    for (const auto& expr : exprs)
      result = SymbolicDimExpr::buildMulExpr(result, expr);
    return result;
  };
  for (int outIdx = outStartIdx; outIdx < outExprs.size(); ++outIdx) {
    subOutExprs.push_back(outExprs[outIdx]);
    subOutDims.push_back(outDims[outIdx]);
    subInExprs.clear();
    subInDims.clear();
    SymbolicDimExpr outProduct = MergeExprs(subOutExprs);
    for (int inIdx = inStartIdx; inIdx < inExprs.size(); ++inIdx) {
      subInExprs.push_back(inExprs[inIdx]);
      subInDims.push_back(inDims[inIdx]);
      SymbolicDimExpr inProduct = MergeExprs(subInExprs);
      if (SymbolicDimExpr::isEqual(outProduct, inProduct)) {
        if (failed(mgr_.mapSymbolicDimProductEqual(
                SymbolicDimProduct{subOutDims},
                SymbolicDimProduct{subInDims}))) {
          return reshapeOp->emitError()
                 << "fail to map partial product equal between the operand "
                    "and result of reshape op\n";
        }
        return tryToBuildProductEqualityForReshape(
            reshapeOp, inExprs, inIdx + 1, outExprs, outIdx + 1);
      }
    }
  }
  return success();
}

// match:
//   %0 = disc_shape.tie_shape %in, %in_d0, %in_d1, ...
//   %out = mhlo.dynamic_reshape(%0, %target_shape) ...
//   %1 = disc_shape.tie_shape %out, %out_d0, %out_d1, ...
// and try to build product-equality between (%in_d0, %in_d1, ...) and
// (%out_d0, %out_d1, ...).
LogicalResult
ShapeComputationIRAnalysis::applyTieShapeOfReshapePatternConstraint(
    Operation* op) {
  auto outTieShape = dyn_cast<disc_shape::TieShapeOp>(op);
  if (!outTieShape) return success();
  auto reshapeOp = dyn_cast_or_null<mhlo::DynamicReshapeOp>(
      outTieShape->getOperand(0).getDefiningOp());
  if (!reshapeOp) return success();
  auto inTieShape = dyn_cast_or_null<disc_shape::TieShapeOp>(
      reshapeOp->getOperand(0).getDefiningOp());
  if (!inTieShape) return success();
  SmallVector<SymbolicDimExpr> inExprs, outExprs;
  llvm::for_each(inTieShape->getOperands().drop_front(),
                 [&](Value v) { inExprs.push_back(value2DefiningExpr_[v]); });
  llvm::for_each(outTieShape->getOperands().drop_front(),
                 [&](Value v) { outExprs.push_back(value2DefiningExpr_[v]); });

  return tryToBuildProductEqualityForReshape(reshapeOp, inExprs, 0, outExprs,
                                             0);
}

LogicalResult ShapeComputationIRAnalysis::applyTieShapeOpConstraint(
    Operation* op) {
  // supppose:
  //   %out = disc_shape.tie_shape(%in, %d0, %d1, ...)
  // we have
  //   - %in and %out should have same shape
  //   - the shape of %out == [d0, d1, ...]
  if (auto tieShape = dyn_cast<disc_shape::TieShapeOp>(op)) {
    if (failed(mapRankedValueShapeEqual(op->getOperand(0), op->getResult(0))))
      return op->emitError()
             << "fail to map the shape of input and output of tie shape op\n";

    auto& resultDims = rankedTensor2SymDims_[op->getResult(0)];
    if (resultDims.size() + 1 != op->getNumOperands())
      return op->emitError()
             << "miss match number shape operand and the rank of result\n";
    for (const auto& en : llvm::enumerate(op->getOperands().drop_front())) {
      if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[en.value()],
                                          resultDims[en.index()])))
        return op->emitError() << "fail to merge symbolic dim\n";
      mgr_.getRootSymbolicDim(resultDims[en.index()])
          .updateKnownNonNegative(true);
    }

    if (isCandidateShapeTensorType(op->getResult(0).getType())) {
      auto& lhsShapeTensorDims = shapeTensor2SymDims_[op->getOperand(0)];
      auto& rhsShapeTensorDims = shapeTensor2SymDims_[op->getResult(0)];
      if (lhsShapeTensorDims.size() != rhsShapeTensorDims.size())
        return op->emitError()
               << "miss match input/output dim size for tie_shape op\n";
      for (const auto& z : llvm::zip(lhsShapeTensorDims, rhsShapeTensorDims)) {
        if (failed(mgr_.mapSymbolicDimEqual(std::get<0>(z), std::get<1>(z))))
          return op->emitError() << "fail to merge dim\n";
      }
    }

    if (failed(applyTieShapeOfReshapePatternConstraint(op)))
      return op->emitError() << "fail to apply tie_shape + reshape like op "
                                "pattern constraint\n";
  }
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyOpConstraint(Operation* op) {
  if (failed(applyIndexOpConstraint(op)))
    return op->emitError() << "fail to apply constraint for index op\n";

  if (failed(applyShapeTensorOpConstraint(op)))
    return op->emitError() << "fail to apply constraint for shape tensor op\n";

  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo") ||
      op->getDialect() == op->getContext()->getLoadedDialect("mhlo_disc")) {
    if (failed(applyMhloOpConstraint(op)))
      return op->emitError() << "fail to apply constraint for mhlo op\n";
  }

  if (failed(applyRankedTensorOpConstraint(op))) {
    return op->emitError() << "fail to apply constraint for ranked tensor op\n";
  }

  if (failed(applyTieShapeOpConstraint(op))) {
    return op->emitError() << "fail to apply constraint for tie_shape op\n";
  }

  return success();
}

DenseMap<SymbolicDimOp, SmallVector<Value>>
ShapeComputationIRAnalysis::getSymbolicDimSSAValueInstance() {
  DenseMap<SymbolicDimOp, SmallVector<Value>> instanceMap;
  funcOp_.walk([&](disc_shape::TieShapeOp tieShapeOp) {
    Value rankedValue = tieShapeOp->getOperand(0);
    auto& symbolicDims = rankedTensor2SymDims_[rankedValue];
    for (auto& en : llvm::enumerate(tieShapeOp->getOperands().drop_front())) {
      SymbolicDimOp root = mgr_.getRootSymbolicDim(symbolicDims[en.index()]);
      instanceMap[root].push_back(en.value());
    }
  });
  return instanceMap;
}

Type ShapeComputationIRAnalysis::getRefinedType(Value value) {
  auto ty = value.getType().dyn_cast<RankedTensorType>();
  if (!ty) return value.getType();

  SmallVector<int64_t> newShape;
  SmallVector<Attribute> refAttrs;
  bool noDynamicDim = true;
  for (SymbolicDimOp sym : rankedTensor2SymDims_[value]) {
    auto root = mgr_.getRootSymbolicDim(sym);
    newShape.push_back(root.getDimSize());
    if (newShape.back() == ShapedType::kDynamic) noDynamicDim = false;
    refAttrs.push_back(SymbolRefAttr::get(value.getContext(), root.getName()));
  }

  if (noDynamicDim) {
    // static shape ranked tensor type: no needs to add symbolic dim ref attrs.
    value.setType(RankedTensorType::get(ty.getShape(), ty.getElementType()));
    return RankedTensorType::get(newShape, ty.getElementType());
  } else {
    auto symbolicShapeAttr = ArrayAttr::get(value.getContext(), refAttrs);
    value.setType(RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                        symbolicShapeAttr));
    return RankedTensorType::get(newShape, ty.getElementType(),
                                 symbolicShapeAttr);
  }
}

SymbolicDimOp ShapeComputationIRAnalysis::value2SymbolicDimOp(Value value) {
  auto it = value2SymDim_.find(value);
  if (it == value2SymDim_.end()) return {};
  return mgr_.getRootSymbolicDim(it->second);
}

llvm::Optional<SmallVector<SymbolicDimOp>>
ShapeComputationIRAnalysis::rankedTensor2SymDims(Value value) {
  auto it = rankedTensor2SymDims_.find(value);
  if (it == rankedTensor2SymDims_.end()) return llvm::None;
  SmallVector<SymbolicDimOp> dims;
  for (SymbolicDimOp dim : it->second)
    dims.push_back(mgr_.getRootSymbolicDim(dim));
  return dims;
}

DenseMap<Value, Value> buildSymbolDimInstancesDominantMap(
    DenseMap<SymbolicDimOp, SmallVector<Value>>& instanceMap,
    DominanceInfo& dominanceInfo) {
  DenseMap<Value, Value> dominantMap;
  for (auto& it : instanceMap) {
    auto& instances = it.second;

    // in normal cases, there should be only one root, aka, dominant value
    // of all the values of instances
    SmallVector<Value> roots;
    for (Value v : instances) {
      bool is_root = true;
      for (Value other : instances) {
        if (v == other) continue;
        if (dominanceInfo.dominates(other, v.getDefiningOp())) {
          is_root = false;
          continue;
        }
      }
      if (is_root) {
        roots.push_back(v);
      }
    }
    // we should let as much values as possible to be dominated by a same root
    for (Value root : roots) {
      for (Value v : instances) {
        if (dominantMap.find(v) == dominantMap.end() &&
            dominanceInfo.dominates(root, v.getDefiningOp())) {
          dominantMap[v] = root;
        }
      }
    }
  }
  return dominantMap;
}

LogicalResult useSameSSAValueIfSymbolicEqual(
    ShapeComputationIRAnalysis& analysis, bool& changed) {
  auto instanceMap = analysis.getSymbolicDimSSAValueInstance();
  DominanceInfo dominanceInfo(analysis.getFunc());
  auto dominantMap =
      buildSymbolDimInstancesDominantMap(instanceMap, dominanceInfo);

  for (auto& pair : dominantMap) {
    Value v = pair.first;
    Value dominant = pair.second;
    if (v != dominant) {
      changed = true;
      v.replaceAllUsesWith(dominant);
    }
  }

  return success();
}

LogicalResult refineTensorType(ShapeComputationIRAnalysis& analysis,
                               bool& changed) {
  auto updateIfNotSame = [&](Value value) {
    Type refinedTy = analysis.getRefinedType(value);
    if (refinedTy != value.getType()) {
      changed = true;
      value.setType(refinedTy);
    }
  };

  FuncOp func = analysis.getFunc();

  // apply refined type for each value
  func.walk([&](Operation* op) {
    for (Value operand : op->getOperands()) updateIfNotSame(operand);

    for (Value result : op->getResults()) updateIfNotSame(result);
  });

  // apply refined function type
  // 1, collect input types
  SmallVector<Type, 4> refinedInputTypes;
  for (Value arg : func.getArguments()) {
    refinedInputTypes.push_back(analysis.getRefinedType(arg));
    if (arg.getType() != refinedInputTypes.back()) changed = true;
  }

  // 2, collect output types
  SmallVector<Type, 4> refinedOutputTypes;
  assert(func.getBody().getBlocks().size() == 1);
  Operation& op = func.getBody().front().getOperations().back();
  for (Value operand : op.getOperands()) {
    refinedOutputTypes.push_back(analysis.getRefinedType(operand));
    if (operand.getType() != refinedOutputTypes.back()) changed = true;
  }

  // 3, refine function type to new type
  auto newFuncTy = FunctionType::get(func.getContext(), refinedInputTypes,
                                     refinedOutputTypes);
  func.setType(newFuncTy);
  return success();
}

LogicalResult tryToSimplifyCompareOp(ShapeComputationIRAnalysis& analysis,
                                     bool& changed) {
  FuncOp func = analysis.getFunc();

  SmallVector<arith::CmpIOp> ops;
  func.walk([&](arith::CmpIOp op) { ops.push_back(op); });

  for (arith::CmpIOp op : ops) {
    OpBuilder b(op);
    auto lhsSym = analysis.value2SymbolicDimOp(op.getLhs());
    auto rhsSym = analysis.value2SymbolicDimOp(op.getRhs());
    if (!lhsSym || !rhsSym) continue;

    Value pred;
    Value truePred = b.create<arith::ConstantIntOp>(op.getLoc(), 1, 1);
    Value falsePred = b.create<arith::ConstantIntOp>(op.getLoc(), 0, 1);
    if (op.getPredicate() == arith::CmpIPredicate::eq) {
      if (lhsSym == rhsSym) {
        pred = truePred;
      } else if (lhsSym.getKnownNonNegative() && rhsSym.getKnownNegativeOne()) {
        pred = falsePred;
      } else if (rhsSym.getKnownNonNegative() && lhsSym.getKnownNegativeOne()) {
        pred = falsePred;
      } else if (lhsSym.getKnownNonSizeZero() && rhsSym.getDimSize() == 0) {
        pred = falsePred;
      } else if (rhsSym.getKnownNonSizeZero() && lhsSym.getDimSize() == 0) {
        pred = falsePred;
      }
      // TODO(disc): support other cases
    } else if (op.getPredicate() == arith::CmpIPredicate::ne) {
      if (lhsSym == rhsSym) {
        pred = falsePred;
      } else if (lhsSym.getKnownNonNegative() && rhsSym.getKnownNegativeOne()) {
        pred = truePred;
      } else if (rhsSym.getKnownNonNegative() && lhsSym.getKnownNegativeOne()) {
        pred = truePred;
      } else if (lhsSym.getKnownNonSizeZero() && rhsSym.getDimSize() == 0) {
        pred = truePred;
      } else if (rhsSym.getKnownNonSizeZero() && lhsSym.getDimSize() == 0) {
        pred = truePred;
      }
      // TODO(disc): support other cases
    }
    // TODO(disc): support arith::CmpIPredicate::lt/...
    if (pred) {
      op.getResult().replaceAllUsesWith(pred);
      changed = true;
    }
  }
  return success();
}

LogicalResult simplifyAccordingToShapeConstraintInfo(
    ShapeComputationIRAnalysis& analysis, bool& changed) {
  if (failed(tryToSimplifyCompareOp(analysis, changed))) return failure();

  // TODO(disc): add other possible simplifier here.
  return success();
}

llvm::Optional<SmallVector<llvm::Optional<int64_t>>>
getConstantElementsOfShapeTensor(Value v) {
  SmallVector<llvm::Optional<int64_t>> results;

  // skip TieShapeOp if necessary.
  while (auto definingOp = v.getDefiningOp<disc_shape::TieShapeOp>()) {
    v = definingOp->getOperand(0);
  }

  // in case the shape tensor is just a constant
  DenseIntElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr))) {
    for (const auto& elem : denseAttr.getValues<APInt>())
      results.push_back(elem.getSExtValue());
    return results;
  }

  // Not known source of shape tensor
  Operation* definingOp = v.getDefiningOp<tensor::FromElementsOp>();
  if (!definingOp) return llvm::None;

  for (Value v : definingOp->getOperands()) {
    auto indexOp = v.getDefiningOp<arith::ConstantOp>();
    if (!indexOp) {
      results.emplace_back(llvm::None);
    } else {
      results.emplace_back(indexOp.getValue().cast<IntegerAttr>().getInt());
    }
  }
  return results;
}

LogicalResult injectStaticKnownInfo(ShapeComputationIRAnalysis& analysis,
                                    bool& changed) {
  SmallVector<mhlo::RealDynamicSliceOp> sliceOps;
  SmallVector<mhlo::DynamicPadOp> padOps;
  analysis.getFunc().walk([&](Operation* op) {
    if (auto sliceOp = dyn_cast<mhlo::RealDynamicSliceOp>(op))
      sliceOps.push_back(sliceOp);
    if (auto padOp = dyn_cast<mhlo::DynamicPadOp>(op)) padOps.push_back(padOp);
  });

  auto tryUpdateStaticKnownInfo =
      [&](Value v, std::function<LogicalResult(int, int64_t)> action) {
        auto values = getConstantElementsOfShapeTensor(v);
        if (!values) return success();
        for (const auto& en : llvm::enumerate(*values)) {
          if (!en.value()) continue;
          if (failed(action(en.index(), *en.value()))) return failure();
        }
        return success();
      };

  for (mhlo::RealDynamicSliceOp op : sliceOps) {
    SliceOpShapeHelper helper(op);
    Value in = op->getOperand(0);
    Value out = op->getResult(0);

    // Check if some axes of the slice are acutally fully selected
    auto inDims = analysis.rankedTensor2SymDims(in);
    auto outDims = analysis.rankedTensor2SymDims(out);
    if (inDims && outDims) {
      for (const auto& en : llvm::enumerate(llvm::zip(*inDims, *outDims))) {
        if (std::get<0>(en.value()) == std::get<1>(en.value()))
          if (failed(helper.markAsFullySlicedAxis(en.index())))
            return op->emitError() << "failed to mark axis" << en.index()
                                   << " to be fully sliced\n";
      }
    }

    // Check if some start/limit/strides are acutally constants
    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(1), [&](int axis, int64_t v) {
              return helper.mergeStartIndex(axis, v);
            })))
      return op->emitError() << "failed to update start index of slice\n";

    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(2), [&](int axis, int64_t v) {
              return helper.mergeLimitIndex(axis, v);
            })))
      return op->emitError() << "failed to update limit index of slice\n";

    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(3),
            [&](int axis, int64_t v) { return helper.mergeStride(axis, v); })))
      return op->emitError() << "failed to update stride of slice\n";

    if (failed(helper.save()))
      return op->emitError() << "failed to save update info for slice op\n";
  }

  for (mhlo::DynamicPadOp op : padOps) {
    PadOpShapeHelper helper(op);
    // Check if some low/high/interior are acutally constants
    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(2), [&](int axis, int64_t v) {
              return helper.mergeEdgePaddingLow(axis, v);
            })))
      return op->emitError() << "failed to update low index of pad\n";

    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(3), [&](int axis, int64_t v) {
              return helper.mergeEdgePaddingHigh(axis, v);
            })))
      return op->emitError() << "failed to update high index of pad\n";

    if (failed(tryUpdateStaticKnownInfo(
            op->getOperand(4), [&](int axis, int64_t v) {
              return helper.mergeInteriorPadding(axis, v);
            })))
      return op->emitError() << "failed to update interior of op\n";

    if (failed(helper.save()))
      return op->emitError() << "failed to save update info for op op\n";
  }

  return success();
}

LogicalResult applyShapeComputationOptimization(
    ShapeComputationIRAnalysis& analysis, bool& changed) {
  // 1, using the same ssa value for all symbolic-equal dim instances.
  if (failed(useSameSSAValueIfSymbolicEqual(analysis, changed)))
    return analysis.getFunc()->emitError(
        "useSameSSAValueIfSymbolicEqual failed\n");

  // 2, After propagation some (partial) known dim size infos, refined
  // the ranked tensor type.
  if (failed(refineTensorType(analysis, changed)))
    return analysis.getFunc()->emitError("refineTensorType failed\n");

  // 3, simplify some expression after propagation shape constraint info.
  // e.g. if symbolic dim %d is known not negative, then `arith.cmpi eq, %d,
  // %c-1` could be replaced with a const.
  if (failed(simplifyAccordingToShapeConstraintInfo(analysis, changed)))
    return analysis.getFunc()->emitError("fail to simplify\n");

  // 4, inject some static known infos. For example,
  // - some axes of a slice op is fully sliced;
  // - some axes of a pad op are not padded;
  if (failed(injectStaticKnownInfo(analysis, changed)))
    return analysis.getFunc()->emitError("fail to injectStaticKnownInfo\n");

  return success();
}

// It’s an iterative process and each iteration can be divided into five steps:
// - Step #1: canonicalization.
//   including normal canonicalization pattern, cse and a bunch of new rewrite
//   patterns (e.g. scalarize shape computation IR whenever possible).
// - Step #2: load existing shape constraint IR
// - Step #3: global shape computation IR analysis.
// - Step #4: gloabl shape computation IR optimization.
// - Step #5: save updated shape constraint IR.
LogicalResult optimizeShapeComputation(ModuleOp m, FuncOp main,
                                       PassPipelineRunner runner) {
  bool changed;
  do {
    changed = false;
    std::chrono::steady_clock::time_point begin, end;
    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    if (failed(runCanonicalizer(m, runner))) {
      return failure();
    }
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  runCanonicalizer takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    LLVM_DEBUG(
        llvm::dbgs()
        << "Module after runCanonicalizer in optimize-shape-computation:\n"
        << m << "\n");

    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    SymbolicDimMgr mgr(m);
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  Building SymbolicDimMgr takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    if (failed(mgr.load())) {
      return m.emitError() << "fail to load shape constraint IR\n";
    }
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  SymbolicDimMgr.load() takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    ShapeComputationIRAnalysis analysis(main, mgr);
    if (failed(analysis.run())) {
      return m.emitError() << "fail to analysis shape computation IR\n";
    }
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  Building ShapeComputationIRAnalysis takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    if (failed(applyShapeComputationOptimization(analysis, changed))) {
      return m.emitError() << "fail to optimize shape computation IR\n";
    }
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  applyShapeComputationOptimization takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    LLVM_DEBUG(
        llvm::dbgs()
        << "Module after apply-shape-opt in optimize-shape-computation:\n"
        << m << "\n");

    DISC_DEBUG(begin = std::chrono::steady_clock::now());
    if (failed(mgr.save())) {
      return m.emitError() << "fail to save shape constraint IR\n";
    }
    DISC_DEBUG(end = std::chrono::steady_clock::now());
    DISC_DEBUG(llvm::dbgs()
               << "  SymbolicDimMgr.save() takes: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                      .count()
               << " us\n");

    LLVM_DEBUG(llvm::dbgs()
               << "Module after save-shape-ir in optimize-shape-computation:\n"
               << m << "\n");
  } while (changed);

  if (failed(runCanonicalizer(m, runner))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after optimizeShapeComputation:\n"
                          << m << "\n");
  return success();
}

/////////////////////////// Stage #2 END //////////////////////////////////////

/////////////////////////// Stage #3 BEGIN ////////////////////////////////////

struct ForwardTieShapeOperandToItsConsumers
    : public OpRewritePattern<disc_shape::TieShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(disc_shape::TieShapeOp tieShapeOp,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(tieShapeOp, tieShapeOp->getOperand(0));
    return success();
  }
};

LogicalResult cleanUp(ModuleOp m, bool keep_tie_shape) {
  if (!keep_tie_shape) {
    RewritePatternSet patterns(m.getContext());
    patterns.add<ForwardTieShapeOperandToItsConsumers>(patterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
      return m.emitError() << "fail to do cleanup\n";
    }
  } else {
    // We will keep a `disc_shape.tie_shape` for each ranked tensor type value.
    // And thus we can safely: 1, drop all symbolicDim reference attributes in
    // ranked tensor type to prepare for bufferization (currently bufferizaiton
    // pass does not support symbolicDim reference). 2, add an attribute to
    // `disc_shape.tie_shape` op to record symbolicDim reference attribute.
    // Example. convert from:
    // ```
    //   %1 = mhlo.abs(%0) : (tensor<?x?xf32, [@S0, @S1]>) -> tensor<?x?xf32,
    //   [@S0, @S1]>
    //   %2 = disc_shape.tie_shape(%1, %d0, %d1) : tensor<?x?xf32, [@S0, @S1]>
    // ```
    // to
    // ```
    //   %1 = mhlo.abs(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    //   %2 = disc_shape.tie_shape(%1, %d0, %d1) {kDiscSymbolicDimAttr = [@S0,
    //   @S1]} : tensor<?x?xf32>
    // ```

    // Part #1: attach the symbolicDim reference attribute to each
    // `disc_shape.tie_shape` op.
    SmallVector<disc_shape::TieShapeOp> tieShapeOps;
    m.walk([&](disc_shape::TieShapeOp op) {
      auto ty = op->getResult(0).getType().dyn_cast<RankedTensorType>();
      if (!ty) return;
      auto attrs = ty.getEncoding().dyn_cast_or_null<ArrayAttr>();
      if (!attrs) return;
      op->setAttr(disc_shape::SymbolicDimOp::getSymbolicDimAttrName(), attrs);
      tieShapeOps.push_back(op);
    });

    for (disc_shape::TieShapeOp op : tieShapeOps) {
      Operation* definingOp = op->getOperand(0).getDefiningOp();
      if (!definingOp) continue;
      if (definingOp->getBlock() != op->getBlock())
        return op->emitError(
            "tie_shape op and the defining op of its source tensor are not in "
            "the same block\n");
      // Try to move the definingOp as close to op as possible.
      SmallVector<Operation*> notUseDefiningOpVec;
      DenseSet<Value> consumersOfDefiningOpSet{definingOp->getResults().begin(),
                                               definingOp->getResults().end()};
      for (auto it = Block::iterator(definingOp), end = Block::iterator(op);
           it != end; ++it) {
        if (llvm::any_of(it->getOperands(), [&](Value val) {
              return consumersOfDefiningOpSet.count(val);
            })) {
          for (Value val : it->getResults())
            consumersOfDefiningOpSet.insert(val);
        } else {
          notUseDefiningOpVec.push_back(&*it);
        }
      }
      for (Operation* op : notUseDefiningOpVec) op->moveBefore(definingOp);
    }

    if (failed(walkRankedTensorValue(
            m, [&](Value value, RankedTensorType ty, ArrayAttr attrs) {
              value.setType(
                  RankedTensorType::get(ty.getShape(), ty.getElementType()));
              return success();
            }))) {
      return failure();
    }
    if (failed(updateFunctionType(m))) return failure();
  }
  return success();
}

/////////////////////////// Stage #3 END //////////////////////////////////////

struct DiscShapeOptimizationPass
    : public DiscShapeOptimizationPassBase<DiscShapeOptimizationPass> {
  DiscShapeOptimizationPass(const std::string& entry_func_name,
                            bool keep_tie_shape)
      : DiscShapeOptimizationPassBase<
            DiscShapeOptimizationPass>::DiscShapeOptimizationPassBase() {
    this->entry_func_name_ = entry_func_name;
    this->keep_tie_shape_ = keep_tie_shape;
  }

  void runOnOperation() override;
};

void DiscShapeOptimizationPass::runOnOperation() {
  ModuleOp m = getOperation();
  FuncOp main = m.lookupSymbol<FuncOp>(entry_func_name_);
  if (!main) {
    m.emitError("entry func: " + entry_func_name_ + " not found");
    signalPassFailure();
    return;
  }

  std::chrono::steady_clock::time_point begin, end;
  DISC_DEBUG(begin = std::chrono::steady_clock::now());
  // Stage #1: Explictily materialize shape computation IR on tensor level
  if (failed(materializeShapeComputation(m, main))) {
    signalPassFailure();
    return;
  }
  DISC_DEBUG(end = std::chrono::steady_clock::now());
  DISC_DEBUG(
      llvm::dbgs() << "materializeShapeComputation takes: "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          end - begin)
                          .count()
                   << " us\n");
  LLVM_DEBUG(llvm::dbgs() << "Module after materialize shape computation:\n"
                          << m << "\n");

  // Stage #2: Optimize shape computation IR on tensor level
  DISC_DEBUG(begin = std::chrono::steady_clock::now());
  PassPipelineRunner runner = [this](OpPassManager& dynamicPM, ModuleOp m) {
    return runPipeline(dynamicPM, m);
  };
  if (failed(optimizeShapeComputation(m, main, runner))) {
    signalPassFailure();
    return;
  }
  DISC_DEBUG(end = std::chrono::steady_clock::now());
  DISC_DEBUG(
      llvm::dbgs() << "optimizeShapeComputation takes: "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          end - begin)
                          .count()
                   << " us\n");
  DISC_DEBUG(llvm::dbgs() << "Module after shape optimizaiton:\n" << m << "\n");

  // Stage #3: clean up
  DISC_DEBUG(begin = std::chrono::steady_clock::now());
  if (failed(cleanUp(m, keep_tie_shape_))) {
    signalPassFailure();
    return;
  }
  DISC_DEBUG(end = std::chrono::steady_clock::now());
  DISC_DEBUG(
      llvm::dbgs() << "cleanUp takes: "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          end - begin)
                          .count()
                   << " us\n");
  DISC_DEBUG(llvm::dbgs() << "Module after cleanup:\n" << m << "\n");
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeOptimizationPass(
    const std::string& entry_func_name, bool keep_tie_shape) {
  return std::make_unique<DiscShapeOptimizationPass>(entry_func_name,
                                                     keep_tie_shape);
}

}  // namespace disc_ral
}  // namespace mlir
