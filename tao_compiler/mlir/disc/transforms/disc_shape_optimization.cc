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
#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"

// #undef LLVM_DEBUG

// #define LLVM_DEBUG(x) (x)

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
    OpResult dimValue = dimOp.source().template dyn_cast<OpResult>();
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
// interface ShapedTypeOpInterface: before
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

  if (failed(
          applyPatternsAndFoldGreedily(m->getRegions(), std::move(patterns)))) {
    return m.emitError() << "fail to materialize shape computation\n";
  }
  return success();
}

/////////////////////////// Stage #1 END //////////////////////////////////////

/////////////////////////// Stage #2 BEGIN ////////////////////////////////////

using PassPipelineRunner =
    std::function<LogicalResult(OpPassManager&, ModuleOp)>;

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
    auto tieShapeOp = op.source().getDefiningOp<disc_shape::TieShapeOp>();
    if (!tieShapeOp) return failure();
    Optional<int64_t> dimIndex = op.getConstantIndex();
    if (!dimIndex) return failure();
    rewriter.replaceOp(op, tieShapeOp->getOperand(1 + *dimIndex));
    return success();
  }
};

// convert:
//   %1 = disc_shape.tie_shape %0, %d0, %d1, ... : (tensor<?x?xf32>, ...) ->
//   tensor<?x?xf32> %dim_size = tensor.dim %1[%c0] : tensor<2xindex>
//   %2 = tensor.extract %1[...] : ...
// to:
//   %2 = tensor.extract %0[...] : ...
struct ExtractElementOfTieShapeOpCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tieShapeOp = op.tensor().getDefiningOp<disc_shape::TieShapeOp>();
    if (!tieShapeOp) return failure();
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, tieShapeOp->getOperand(0), op.indices());
    return success();
  }
};

// Adds shape optimization related patterns.
void populateShapeOptimizationPatterns(MLIRContext* context,
                                       RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      DimOfTieShapeOpCanonicalizationPattern,
      ExtractElementOfTieShapeOpCanonicalizationPattern
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

  if (failed(applyPatternsAndFoldGreedily(m->getRegions(),
                                          std::move(frozenSet)))) {
    return m.emitError() << "fail to run canonicalizer\n";
  }

  OpPassManager dynamicPM("builtin.func");
  dynamicPM.addPass(createCSEPass());
  return runner(dynamicPM, m);
}

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
  } else if (auto tensorTy = ty.dyn_cast<RankedTensorType>()) {
    SmallVector<SymbolicDimOp> symbols =
        mgr_.getOrCreateSymbolicDimsForRankedValue(value);
    rankedTensor2SymDims_[value] = std::move(symbols);
    // Try to check if it's a candidate shape tensor.
    if (isCandidateShapeTensorType(ty)) {
      SmallVector<SymbolicDimOp> symbols;
      for (int i = 0, d = tensorTy.getShape()[0]; i < d; ++i)
        symbols.push_back(mgr_.newSymbolicDim());
      shapeTensor2SymDims_[value] = symbols;
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

  if (isa<arith::IndexCastOp>(op)) {
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[op->getResult(0)],
                                        value2SymDim_[op->getOperand(0)])))
      return op->emitError() << "fail to merge dim\n";
  } else if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex) return success();
    // TODO: set isKnownNonNegative
    if (failed(mgr_.mapSymbolicDimEqual(
            value2SymDim_[op->getResult(0)],
            rankedTensor2SymDims_[dimOp.source()][*dimIndex])))
      return op->emitError() << "fail to merge dim\n";
  } else if (isa<arith::ConstantIndexOp, arith::ConstantIntOp>(op)) {
    int64_t val = op->getAttrOfType<IntegerAttr>("value").getInt();
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[op->getResult(0)],
                                        mgr_.newConstantSymbolicDim(val))))
      return op->emitError() << "fail to merge dim\n";
  } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (!isCandidateShapeTensorType(extractOp.tensor().getType()))
      return success();
    auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(
        op->getOperand(1).getDefiningOp());
    if (!indexOp) return success();
    int64_t index = indexOp.getValue().cast<IntegerAttr>().getInt();
    auto& shapeTensorDims = shapeTensor2SymDims_[extractOp.tensor()];
    if (index >= shapeTensorDims.size())
      return op->emitError() << "miss match shape tensor size\n";
    if (failed(mgr_.mapSymbolicDimEqual(value2SymDim_[op->getResult(0)],
                                        shapeTensorDims[index])))
      return op->emitError() << "fail to merge dim\n";
  }

  // TODO: add support for arith::addi/subi/...

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyShapeTensorOpConstraint(
    Operation* op) {
  if (isa<tensor::FromElementsOp>(op)) {
    auto& symbols = shapeTensor2SymDims_[op->getResult(0)];
    if (symbols.size() != op->getOperands().size())
      op->emitError() << "miss match dim size and num operands\n";
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
  }

  // TODO: add support for arith::addi/subi/...

  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyMhloOpConstraint(Operation* op) {
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
  } else if (isa<mhlo::DynamicBroadcastInDimOp>(op)) {
    auto& shapeTensorDims = shapeTensor2SymDims_[op->getOperand(1)];
    auto& outDims = rankedTensor2SymDims_[op->getResult(0)];

    if (shapeTensorDims.size() != outDims.size())
      return op->emitError() << "mismatch out rank and shape tensor size\n";
    for (const auto& z : llvm::zip(shapeTensorDims, outDims)) {
      if (failed(mgr_.mapSymbolicDimEqual(std::get<0>(z), std::get<1>(z))))
        return op->emitError() << "fail to merge dim\n";
    }
  } else if (auto concat = dyn_cast<mhlo::ConcatenateOp>(op)) {
    Value out = op->getResult(0);
    int64_t axis = concat.dimension();
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
  } else if (auto dot_general = dyn_cast<mhlo::DotGeneralOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return success();
    auto& lhsDims = rankedTensor2SymDims_[lhs];
    auto& rhsDims = rankedTensor2SymDims_[rhs];
    if (lhsTy.getRank() != lhsDims.size() || rhsTy.getRank() != rhsDims.size())
      return op->emitError("lhs or rhs mismatch rank\n");
    auto dim_numbers = dot_general.dot_dimension_numbers();
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

    if (failed(mgr_.mapSymbolicDimEqual(lhsDims[1], rhsDims[0])))
      return op->emitError() << "fail to merge dim\n";
  } else if (auto clamp = dyn_cast<mhlo::ClampOp>(op)) {
    auto operandTy = clamp.operand().getType().dyn_cast<RankedTensorType>();
    auto minTy = clamp.min().getType().dyn_cast<RankedTensorType>();
    auto maxTy = clamp.max().getType().dyn_cast<RankedTensorType>();
    if (!operandTy || !minTy || !maxTy) return success();

    if (minTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(clamp.operand(), clamp.min())))
        return op->emitError()
               << "fail to merge the symbolic dim of operand and "
                  "min of mhlo::ClampOp\n";
    }
    if (maxTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(clamp.operand(), clamp.max())))
        return op->emitError()
               << "fail to merge the symbolic dim of operand and "
                  "max of mhlo::ClampOp\n";
    }
  } else if (auto select = dyn_cast<mhlo::SelectOp>(op)) {
    auto predTy = select.pred().getType().dyn_cast<RankedTensorType>();
    auto trueTy = select.on_true().getType().dyn_cast<RankedTensorType>();
    auto falseTy = select.on_false().getType().dyn_cast<RankedTensorType>();
    auto resultTy = select.getResult().getType().dyn_cast<RankedTensorType>();
    if (!predTy || !trueTy || !falseTy || !resultTy) return success();

    if (predTy.getRank() != 0) {
      if (failed(mapRankedValueShapeEqual(select.pred(), select.getResult())))
        return op->emitError() << "fail to merge the symbolic dim of pred and "
                                  "result of mhlo::SelectOp\n";
    }
    if (trueTy.getRank() != 0) {
      if (failed(
              mapRankedValueShapeEqual(select.on_true(), select.getResult())))
        return op->emitError()
               << "fail to merge the symbolic dim of on_true and "
                  "result of mhlo::SelectOp\n";
    }
    if (falseTy.getRank() != 0) {
      if (failed(
              mapRankedValueShapeEqual(select.on_false(), select.getResult())))
        return op->emitError()
               << "fail to merge the symbolic dim of on_false and "
                  "result of mhlo::SelectOp\n";
    }
  }
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
    if (newShape.back() == ShapedType::kDynamicSize) noDynamicDim = false;
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

    // if (auto constOp = dyn_cast<mhlo::ConstOp>(op)) {
    //   auto attr = constOp.value().cast<DenseElementsAttr>();
    //   auto newAttr = DenseElementsAttr::get(op->getResult(0).getType(),
    //   attr.getRawData()); op->setAttr("value", newAttr);
    // }
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

LogicalResult applyShapeComputationOptimization(
    ShapeComputationIRAnalysis& analysis, bool& changed) {
  // 1, using the same ssa value for all symbolic-equal dim instances.
  if (failed(useSameSSAValueIfSymbolicEqual(analysis, changed)))
    return analysis.getFunc()->emitError(
        "useSameSSAValueIfSymbolicEqual failed");

  // 2, After propagation some (partial) known dim size infos, refined
  // the ranked tensor type.
  if (failed(refineTensorType(analysis, changed)))
    return analysis.getFunc()->emitError("refineTensorType failed");

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
    if (failed(runCanonicalizer(m, runner))) {
      return failure();
    }

    LLVM_DEBUG(
        llvm::dbgs()
        << "Module after runCanonicalizer in optimize-shape-computation:\n"
        << m << "\n");

    SymbolicDimMgr mgr(m);
    if (failed(mgr.load())) {
      return m.emitError() << "fail to load shape constraint IR\n";
    }

    ShapeComputationIRAnalysis analysis(main, mgr);
    if (failed(analysis.run())) {
      return m.emitError() << "fail to analysis shape computation IR\n";
    }

    if (failed(applyShapeComputationOptimization(analysis, changed))) {
      return m.emitError() << "fail to optimize shape computation IR\n";
    }

    LLVM_DEBUG(
        llvm::dbgs()
        << "Module after apply-shape-opt in optimize-shape-computation:\n"
        << m << "\n");

    if (failed(mgr.save())) {
      return m.emitError() << "fail to save shape constraint IR\n";
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Module after save-shape-ir in optimize-shape-computation:\n"
               << m << "\n");

  } while (changed);
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

    if (failed(applyPatternsAndFoldGreedily(m->getRegions(),
                                            std::move(patterns)))) {
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
    //   [@S0, @S1]> %2 = disc_shape.tie_shape(%1, %d0, %d1) : tensor<?x?xf32,
    //   [@S0, @S1]>
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

  // Stage #1: Explictily materialize shape computation IR on tensor level
  if (failed(materializeShapeComputation(m, main))) {
    signalPassFailure();
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after materialize shape computation:\n"
                          << m << "\n");

  // Stage #2: Optimize shape computation IR on tensor level
  PassPipelineRunner runner = [this](OpPassManager& dynamicPM, ModuleOp m) {
    return runPipeline(dynamicPM, m);
  };
  if (failed(optimizeShapeComputation(m, main, runner))) {
    signalPassFailure();
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after shape optimizaiton:\n" << m << "\n");

  // Stage #3: clean up
  if (failed(cleanUp(m, keep_tie_shape_))) {
    signalPassFailure();
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Module after cleanup:\n" << m << "\n");
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeOptimizationPass(
    const std::string& entry_func_name, bool keep_tie_shape) {
  return std::make_unique<DiscShapeOptimizationPass>(entry_func_name,
                                                     keep_tie_shape);
}

}  // namespace disc_ral
}  // namespace mlir
