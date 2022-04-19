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

#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization.h"

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
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

#undef LLVM_DEBUG

#define LLVM_DEBUG(x) (x)

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;

int64_t SymbolicDim::uniqueId() const {
  // TODO
  return 0;
}

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) {
  // TODO
}

LogicalResult SymbolicDimMgr::load() {
  // TODO
  return success();
}

LogicalResult SymbolicDimMgr::save() {
  // TODO
  return success();
}

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
LogicalResult insertTieShapeOps(FuncOp main) {
  // TODO
  return success();
}

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
  if (failed(insertTieShapeOps(main))) {
    return failure();
  }

  RewritePatternSet patterns(m.getContext());
  patterns.add<DimOfShapedTypeOpInterface<tensor::DimOp>>(
      patterns.getContext());

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

// Adds canonicalization patterns to the list of patterns.
void addCanonicalizationPatterns(MLIRContext* context,
                                 RewritePatternSet* patterns) {
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

LogicalResult runCanonicalizer(ModuleOp m, PassPipelineRunner runner) {
  MLIRContext* context = m.getContext();
  RewritePatternSet patterns(context);
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);

  if (failed(
          applyPatternsAndFoldGreedily(m->getRegions(), std::move(patterns)))) {
    return m.emitError() << "fail to run canonicalizer\n";
  }

  OpPassManager dynamicPM("builtin.func");
  dynamicPM.addPass(createCSEPass());
  return runner(dynamicPM, m);
}

class ShapeComputationIRAnalysis {
 public:
  explicit ShapeComputationIRAnalysis(ModuleOp m, SymbolicDimMgr& mgr);

  LogicalResult run();

 private:
  LogicalResult runOnRegion(Region* region);
  LogicalResult runOnBlock(Block* block);
  LogicalResult runOnOperation(Operation* op);
  LogicalResult buildSymbolicShape(Value value);
  LogicalResult applyOpConstraint(Operation* op);

 private:
  bool initialized_ = false;
  ModuleOp moduleOp_;
  SymbolicDimMgr& mgr_;

  // Map scalar int/index SSA value to a symbolicDim
  DenseMap<Value, SymbolicDim*> value2SymDim_;

  // Map a shape tensor value (1D ranked int/index tensor) to an array of
  // symbolicDims, each for one component of the shape tensor.
  DenseMap<Value, SmallVector<SymbolicDim*>> shapeTensor2SymDims_;

  // Map a ranked tensor value to an array of symbolicDims, each represents one
  // dimension size of the tensor.
  DenseMap<Value, SmallVector<SymbolicDim*>> rankedTensor2SymDims_;
};

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(ModuleOp m,
                                                       SymbolicDimMgr& mgr)
    : moduleOp_(m), mgr_(mgr) {}

LogicalResult ShapeComputationIRAnalysis::run() {
  ModuleOp& m = moduleOp_;
  // Make sure only run once.
  if (initialized_) {
    return m->emitError() << "re-initialized shape analysis is not supported\n";
  }
  initialized_ = true;

  for (auto& region : m->getRegions()) {
    if (failed(runOnRegion(&region))) {
      return failure();
    }
  }
  return success();
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
  WalkResult result = block->walk([&](Operation* op) {
    if (failed(runOnOperation(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult ShapeComputationIRAnalysis::runOnOperation(Operation* op) {
  // build shapes for the results of op
  for (Value result : op->getResults()) {
    if (failed(buildSymbolicShape(result))) {
      return op->emitError("failed to build shape for op's result");
    }
  }

  // apply op's shape constraint
  return applyOpConstraint(op);
}

LogicalResult ShapeComputationIRAnalysis::buildSymbolicShape(Value value) {
  return success();
}

LogicalResult ShapeComputationIRAnalysis::applyOpConstraint(Operation* op) {
  return success();
}

LogicalResult applyShapeComputationOptimization(
    ShapeComputationIRAnalysis& analysis, bool& changed) {
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

    SymbolicDimMgr mgr(m);
    if (failed(mgr.load())) {
      return m.emitError() << "fail to load shape constraint IR\n";
    }

    ShapeComputationIRAnalysis analysis(m, mgr);
    if (failed(analysis.run())) {
      return m.emitError() << "fail to analysis shape computation IR\n";
    }

    if (failed(applyShapeComputationOptimization(analysis, changed))) {
      return m.emitError() << "fail to optimize shape computation IR\n";
    }

    if (failed(mgr.save())) {
      return m.emitError() << "fail to save shape constraint IR\n";
    }
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
