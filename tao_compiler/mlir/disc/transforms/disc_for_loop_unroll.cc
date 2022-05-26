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

#include "PassDetail.h"
#include "llvm/ADT/Sequence.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/core/util/env_var.h"

// This pass unrolls the inner-most SCF for-loop ops within the given
// operation/module. It fixes a bug of `loopUnrollByFactor` of llvm-project, and
// clones other parts into this file. We will use the `loopUnrollByFactor`
// function directly when the bugfix is merged into the llvm master branch. Note
// that we do not use patch to fix this bug to avoid the possible works when
// rebasing newer TF commits.

namespace mlir {
namespace disc_ral {

namespace {

bool getInnermostForLoops(Operation* rootOp,
                          SmallVectorImpl<scf::ForOp>& result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesFloops = false;
  for (Region& region : rootOp->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block) {
        bool enclosesFloops = getInnermostForLoops(&op, result);
        rootEnclosesFloops |= enclosesFloops;
        if (auto floop = dyn_cast<scf::ForOp>(op)) {
          rootEnclosesFloops = true;

          // Collect For loop if it is an innermost one.
          if (!enclosesFloops) result.push_back(floop);
        }
      }
    }
  }
  return rootEnclosesFloops;
}

// Build the IR that performs ceil division of a positive value by another
// positive value:
//    ceildiv(a, b) = divis(a + (b - 1), b)
// where divis is rounding-to-zero division.
Value ceilDivPositive(OpBuilder& builder, Location loc, Value dividend,
                      Value divisor) {
  assert(dividend.getType().isIndex() && "expected index-typed value");
  Value cstOne = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value divisorMinusOne = builder.create<arith::SubIOp>(loc, divisor, cstOne);
  Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOne);
  return builder.create<arith::DivSIOp>(loc, sum, divisor);
}

/// Generates unrolled copies of scf::ForOp 'loopBodyBlock', with
/// associated 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'forOpIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
void generateUnrolledLoop(
    Block* loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn) annotateFn = [](unsigned, Operation*, OpBuilder) {};

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  for (unsigned i = 1; i < unrollFactor; i++) {
    BlockAndValueMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      Operation* clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++)
      lastYielded[i] = operandMap.lookup(yieldedValues[i]);
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++)
    annotateFn(0, &*it, builder);

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

/// Unrolls 'forOp' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult loopUnrollByFactor(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  OpBuilder boundsBuilder(forOp);
  auto loc = forOp.getLoc();
  auto step = forOp.getStep();
  Value upperBoundUnrolled;
  Value stepUnrolled;
  bool generateEpilogueLoop = true;

  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = (ubCst - lbCst + stepCst - 1) / stepCst;
    // int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);

    if (unrollFactor == 1) {
      if (tripCount == 1 && failed(promoteIfSingleIteration(forOp)))
        return failure();
      return success();
    }

    int64_t tripCountEvenMultiple = tripCount - (tripCount % unrollFactor);
    int64_t upperBoundUnrolledCst = lbCst + tripCountEvenMultiple * stepCst;
    assert(upperBoundUnrolledCst <= ubCst);
    int64_t stepUnrolledCst = stepCst * unrollFactor;

    // Create constant for 'upperBoundUnrolled' and set epilogue loop flag.
    generateEpilogueLoop = upperBoundUnrolledCst < ubCst;
    if (generateEpilogueLoop)
      upperBoundUnrolled = boundsBuilder.create<arith::ConstantIndexOp>(
          loc, upperBoundUnrolledCst);
    else
      upperBoundUnrolled = ubCstOp;

    // Create constant for 'stepUnrolled'.
    stepUnrolled = stepCst == stepUnrolledCst
                       ? step
                       : boundsBuilder.create<arith::ConstantIndexOp>(
                             loc, stepUnrolledCst);
  } else {
    // Dynamic loop bounds computation.
    // TODO: Add dynamic asserts for negative lb/ub/step, or
    // consider using ceilDiv from AffineApplyExpander.
    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    Value diff =
        boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value tripCount = ceilDivPositive(boundsBuilder, loc, diff, step);
    Value unrollFactorCst =
        boundsBuilder.create<arith::ConstantIndexOp>(loc, unrollFactor);
    Value tripCountRem =
        boundsBuilder.create<arith::RemSIOp>(loc, tripCount, unrollFactorCst);
    // Compute tripCountEvenMultiple = tripCount - (tripCount % unrollFactor)
    Value tripCountEvenMultiple =
        boundsBuilder.create<arith::SubIOp>(loc, tripCount, tripCountRem);
    // Compute upperBoundUnrolled = lowerBound + tripCountEvenMultiple * step
    upperBoundUnrolled = boundsBuilder.create<arith::AddIOp>(
        loc, lowerBound,
        boundsBuilder.create<arith::MulIOp>(loc, tripCountEvenMultiple, step));
    // Scale 'step' by 'unrollFactor'.
    stepUnrolled =
        boundsBuilder.create<arith::MulIOp>(loc, step, unrollFactorCst);
  }

  // Create epilogue clean up loop starting at 'upperBoundUnrolled'.
  if (generateEpilogueLoop) {
    OpBuilder epilogueBuilder(forOp->getContext());
    epilogueBuilder.setInsertionPoint(forOp->getBlock(),
                                      std::next(Block::iterator(forOp)));
    auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
    epilogueForOp.setLowerBound(upperBoundUnrolled);

    // Update uses of loop results.
    auto results = forOp.getResults();
    auto epilogueResults = epilogueForOp.getResults();
    auto epilogueIterOperands = epilogueForOp.getIterOperands();

    for (auto e : llvm::zip(results, epilogueResults)) {
      std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
    }
    for (int64_t i = 0; i < epilogueForOp.getNumIterOperands(); i++) {
      epilogueForOp.getOperation()->setOperand(
          i + epilogueForOp.getNumControlOperands(), results[i]);
    }
    (void)promoteIfSingleIteration(epilogueForOp);
  }

  // Create unrolled loop.
  forOp.setUpperBound(upperBoundUnrolled);
  forOp.setStep(stepUnrolled);

  auto iterArgs = ValueRange(forOp.getRegionIterArgs());
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();

  generateUnrolledLoop(forOp.getBody(), forOp.getInductionVar(), unrollFactor,
                       [&](unsigned i, Value iv, OpBuilder b) {
                         // iv' = iv + step * i;
                         auto stride = b.create<arith::MulIOp>(
                             loc, step,
                             b.create<arith::ConstantIndexOp>(loc, i));
                         return b.create<arith::AddIOp>(loc, iv, stride);
                       },
                       annotateFn, iterArgs, yieldedValues);
  // Promote the loop body up if this has turned into a single iteration loop.
  (void)promoteIfSingleIteration(forOp);
  return success();
}

struct ForLoopUnroll : public ForLoopUnrollBase<ForLoopUnroll> {
  ForLoopUnroll() = default;
  explicit ForLoopUnroll(int64_t unrollFactor) {
    this->unrollFactor = unrollFactor;
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp, 4> loops;
    getInnermostForLoops(getOperation(), loops);
    bool annotateLoop = false;  // set to `true` for debugging.
    auto annotateFn = [&](unsigned i, Operation* op, OpBuilder b) {
      if (annotateLoop) {
        op->setAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
      }
    };
    for (auto loop : loops) {
      auto result =
          disc_ral::loopUnrollByFactor(loop, this->unrollFactor, annotateFn);
      if (failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createForLoopUnrollPass(int64_t unrollFactor) {
  return std::make_unique<ForLoopUnroll>(unrollFactor);
}

}  // namespace disc_ral
}  // namespace mlir