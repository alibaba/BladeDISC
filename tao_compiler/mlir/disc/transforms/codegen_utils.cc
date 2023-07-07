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
#include "mlir/disc/transforms/codegen_utils.h"

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/disc_util.h"

using mlir::memref::DimOp;

namespace mlir {
namespace disc_ral {

int getRowReductionScheduleHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = dyn_cast<lmhlo::FusionOp>(op);
  if (!fusion) {
    fusion = op->getParentOfType<lmhlo::FusionOp>();
  }
  // Use schedule 1 by default.
  // Schedule 1 has a better performance in a wider range of shapes than
  // schedule 2.
  if (!fusion) return DISC_BLOCK_WISE_ROW_REDUCE;
  IntegerAttr attr =
      fusion->getAttrOfType<IntegerAttr>(kRowReductionScheduleHint);
  if (!attr) return DISC_BLOCK_WISE_ROW_REDUCE;
  return attr.getInt();
}

int getVectorizeOrTileHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = dyn_cast<lmhlo::FusionOp>(op);
  if (!fusion) {
    fusion = op->getParentOfType<lmhlo::FusionOp>();
  }
  if (!fusion) {
    return 1;
  }
  IntegerAttr attr = fusion->getAttrOfType<IntegerAttr>(kVectorizeOrTileHint);
  if (!attr) {
    return 1;
  }
  return attr.getInt();
}

int getThreadPerBlock(Operation* op) {
  int thread_per_block = kThreadsRowReduction;
  if (!op) return thread_per_block;
  lmhlo::FusionOp fusion = dyn_cast<lmhlo::FusionOp>(op);
  if (!fusion) {
    fusion = op->getParentOfType<lmhlo::FusionOp>();
  }
  if (!fusion) return thread_per_block;
  IntegerAttr attr = fusion->getAttrOfType<IntegerAttr>(kThreadPerBlockHint);
  if (!attr) return thread_per_block;
  return attr.getInt();
}

int getColReductionScheduleHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = dyn_cast<lmhlo::FusionOp>(op);
  if (!fusion) {
    fusion = op->getParentOfType<lmhlo::FusionOp>();
  }
  // Use schedule 1 by default.
  if (!fusion) return DISC_THREAD_TILE_H32;
  IntegerAttr attr =
      fusion->getAttrOfType<IntegerAttr>(kColReductionScheduleHint);
  if (!attr) return DISC_THREAD_TILE_H32;
  return attr.getInt();
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref) {
  int rank = memref.getType().cast<MemRefType>().getRank();
  Value num_elements;
  num_elements = b.create<arith::ConstantIndexOp>(loc, 1);
  for (int r = 0; r < rank; ++r) {
    auto dim_size = b.create<memref::DimOp>(loc, memref, r);
    num_elements = b.create<arith::MulIOp>(loc, num_elements, dim_size);
  }
  return num_elements;
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op) {
  // only const rank is supported for now
  assert(op->getDialect()->getNamespace() == "lmhlo");
  int num_operands = op->getNumOperands();
  Value result_memref = op->getOperand(num_operands - 1);
  return emitNumElementsComputation(b, loc, result_memref);
}

Value calcLinearIndex(OpBuilder* b, Location loc, const ValueRange multi_index,
                      const llvm::ArrayRef<Value> shape) {
  assert(multi_index.size() == shape.size());
  return b->create<disc_shape::LinearizeOp>(loc, b->getIndexType(), multi_index,
                                            shape);
}

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index,
                                     const ArrayRef<Value> shape) {
  int rank = shape.size();
  SmallVector<Type> outputTypes(rank, b->getIndexType());
  auto delinearizeOp = b->create<disc_shape::DelinearizeOp>(
      loc, outputTypes, linear_index, shape);

  SmallVector<Value> results;
  for (Value result : delinearizeOp->getResults()) results.push_back(result);
  return results;
}

Value getDimSizeValue(OpBuilder* b, Value memref, int dim) {
  auto memref_ty = memref.getType().dyn_cast<ShapedType>();
  auto loc = memref.getLoc();
  assert(memref_ty && memref_ty.getRank() > dim);
  auto dim_size = memref_ty.getDimSize(dim);
  if (dim_size == ShapedType::kDynamic) {
    return b->create<DimOp>(loc, memref, dim);
  } else {
    return b->create<arith::ConstantIndexOp>(loc, dim_size);
  }
}

Value mayConvertToIndexType(Value val, OpBuilder* b, Location loc) {
  if (val.getType().isIndex()) return val;
  return b->create<arith::IndexCastOp>(loc, b->getIndexType(), val);
}

Value mayConvertToIntegerType(Value val, OpBuilder* b, Location loc) {
  if (val.getType().isInteger(32) || val.getType().isInteger(64)) return val;
  return b->create<arith::IndexCastOp>(loc, b->getI64Type(), val);
}

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index, Value memref) {
  SmallVector<Value> shape_vec = getShapeValues(b, memref);
  return calcMultiDimIndex(b, loc, linear_index, shape_vec);
}

using scf::IfOp;
using scf::ParallelOp;

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> init_values) {
  auto par_op = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, init_values,
                                          /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(par_op.getBody());
  vars.append(par_op.getInductionVars().begin(),
              par_op.getInductionVars().end());
  return par_op;
}

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(%arg4*tileSize[0], %arg2-%i0)
///                                          min(%arg5*tileSize[1], %arg3-%i1))
///                                      step (%arg4, %arg5)
///
/// or, when with-inbound-check is true, into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (%arg4*tileSize[0],
///                                          %arg5*tileSize[1])
///                                      step (%arg4, %arg5)
///        %inbound = (%j0 * %arg4 + %i0 < %arg2) &&
///                   (%j1 * %arg5 + %i1 < %arg3)
///        scf.if (%inbound)
///          ....
///
/// where the uses of %i0 and %i1 in the loop body are replaced by
/// %i0 + j0 and %i1 + %j1.
//
/// The old loop is replaced with the new one.
std::pair<ParallelOp, ParallelOp> tileParallelLoop(ParallelOp op,
                                                   ArrayRef<int64_t> tileSizes,
                                                   bool withInboundCheck) {
  OpBuilder b(op);
  auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  SmallVector<Value, 2> tileSizeConstants;
  tileSizeConstants.reserve(op.getUpperBound().size());
  for (size_t i = 0, end = op.getUpperBound().size(); i != end; ++i) {
    if (i < tileSizes.size())
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), tileSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create the outer loop with adjusted steps.
  SmallVector<Value, 2> newSteps;
  newSteps.reserve(op.getStep().size());
  for (auto step : llvm::zip(op.getStep(), tileSizeConstants)) {
    newSteps.push_back(b.create<arith::MulIOp>(op.getLoc(), std::get<0>(step),
                                               std::get<1>(step)));
  }
  auto outerLoop = b.create<ParallelOp>(op.getLoc(), op.getLowerBound(),
                                        op.getUpperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, b.getContext()),
       getAffineDimExpr(/*position=*/1, b.getContext()) -
           getAffineDimExpr(/*position=*/2, b.getContext())},
      b.getContext());

  // Create the inner loop with adjusted bounds.
  SmallVector<Value, 2> newBounds;
  newBounds.reserve(op.getUpperBound().size());
  bool needInboundCheck = false;
  for (auto dim :
       llvm::zip(outerLoop.getLowerBound(), outerLoop.getUpperBound(),
                 outerLoop.getStep(), outerLoop.getInductionVars(),
                 op.getStep(), tileSizeConstants)) {
    Value lowerBound, upperBound, newStep, iv, step, tileSizeConstant;
    std::tie(lowerBound, upperBound, newStep, iv, step, tileSizeConstant) = dim;
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
    auto tileSize =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp())
            .getValue()
            .cast<IntegerAttr>()
            .getInt();
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(
          upperBoundConstant.getValue().cast<IntegerAttr>().getInt() -
              lowerBoundConstant.getValue().cast<IntegerAttr>().getInt(),
          stepConstant.getValue().cast<IntegerAttr>().getInt());
      if (numIterations % tileSize == 0) {
        newBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (withInboundCheck) {
      newBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBounds.push_back(
        b.create<AffineMinOp>(op.getLoc(), b.getIndexType(), minMap,
                              ValueRange{newStep, upperBound, iv}));
  }
  auto innerLoop = b.create<ParallelOp>(
      op.getLoc(), SmallVector<Value, 2>(newBounds.size(), zero), newBounds,
      op.getStep());

  if (withInboundCheck && needInboundCheck) {
    b.setInsertionPointToStart(innerLoop.getBody());
    // Insert in-bound check
    Value inbound =
        b.create<arith::ConstantIntOp>(op.getLoc(), 1, /*bitWidth*/ 1);
    for (auto dim :
         llvm::zip(outerLoop.getUpperBound(), outerLoop.getInductionVars(),
                   innerLoop.getInductionVars(), innerLoop.getStep())) {
      Value outerUpperBound, outerIV, innerIV, innerStep;
      std::tie(outerUpperBound, outerIV, innerIV, innerStep) = dim;
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv < %outer_upper_bound)
      Value index = b.create<arith::AddIOp>(
          op.getLoc(), b.create<arith::MulIOp>(op.getLoc(), innerIV, innerStep),
          outerIV);
      Value dimInbound = b.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ult, index, outerUpperBound);
      inbound = b.create<arith::AndIOp>(op.getLoc(), inbound, dimInbound);
    }
    auto ifInbound = b.create<IfOp>(op.getLoc(),
                                    /*resultTypes*/ ArrayRef<Type>{}, inbound,
                                    /*hasElseRegion*/ false);
    ifInbound.getThenRegion().takeBody(op.getLoopBody());
    Block& thenBlock = ifInbound.getThenRegion().front();
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::enumerate(llvm::zip(innerLoop.getInductionVars(),
                                              outerLoop.getInductionVars()))) {
      arith::AddIOp newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs.value()), std::get<1>(ivs.value()));
      thenBlock.getArgument(ivs.index())
          .replaceAllUsesExcept(newIndex, newIndex);
    }
    thenBlock.eraseArguments(0, thenBlock.getNumArguments());
  } else {
    innerLoop.getLoopBody().takeBody(op.getLoopBody());
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::zip(innerLoop.getInductionVars(),
                              outerLoop.getInductionVars())) {
      Value inner_index = std::get<0>(ivs);
      arith::AddIOp newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs), std::get<1>(ivs));
      inner_index.replaceAllUsesExcept(newIndex, newIndex);
    }
  }

  op.erase();
  return std::make_pair(outerLoop, innerLoop);
}

int initReductionTileSizeOnCPU() {
  const char* env = getenv("DISC_REDUCTION_TILE_SIZE");
  if (!env) return kReductionTileSizeOnCPU;
  return std::stoi(env);
}

int getReductionTileSizeOnCPU() {
  static int size = initReductionTileSizeOnCPU();
  return size;
}

// Get ops that depends on the given op in the same block. It assumes the ops in
// the block do not have regions.
void getDependentOpsInBlock(Operation* op, DenseSet<Operation*>& dependences) {
  SmallVector<Value> affectedValues;
  // TODO: deal with AssumeAlignmentOp.
  for (auto result : op->getResults()) {
    affectedValues.push_back(result);
  }
  for (auto operand : op->getOperands()) {
    if (IsOpWriteValue(op, operand)) {
      affectedValues.push_back(operand);
    }
  }

  auto block = op->getBlock();
  for (auto value : affectedValues) {
    for (auto user : value.getUsers()) {
      if (user == op || user->getBlock() != block ||
          user->isBeforeInBlock(op)) {
        continue;
      }
      dependences.insert(user);
      getDependentOpsInBlock(user, dependences);
    }
  }
}

// Check whether a depends on b.
bool dependsOnInBlock(Operation* a, Operation* b) {
  DenseSet<Operation*> dependences;
  getDependentOpsInBlock(b, dependences);
  return dependences.contains(a);
}

/// Generates unrolled copies of scf::ForOp 'loopBodyBlock', with
/// associated 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'forOpIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn. It tries to interleave the unrolled
// loop.
LogicalResult generateUnrolledLoopMayInterleave(
    Block* loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn) annotateFn = [](unsigned, Operation*, OpBuilder) {};

  // The number of operators excep the terminator of this block.
  int64_t numOps =
      std::distance(loopBodyBlock->begin(), std::prev(loopBodyBlock->end(), 1));

  // If the induction variable is used, create the transformed value for each
  // iteration of this unrolled instance.
  SmallVector<Value> ivs(unrollFactor);
  if (!forOpIV.use_empty()) {
    builder.setInsertionPointToStart(loopBodyBlock);
    for (int64_t i = 1; i < unrollFactor; i++) {
      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      ivs[i] = ivUnroll;
    }
  }

  // The index of the first non-induction-variable transformation instruction,
  // i.e., the index of the first to-be-cloned instruction. It is the same with
  // the number of instructions created when building `ivs`.
  int64_t beginIdx = std::distance(loopBodyBlock->begin(),
                                   std::prev(loopBodyBlock->end(), 1)) -
                     numOps;

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);
  builder.setInsertionPointAfter(&(*srcBlockEnd));

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  for (unsigned i = 1; i < unrollFactor; i++) {
    IRMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      operandMap.map(forOpIV, ivs[i]);
    }

    // Clone the original body of 'forOp'.
    int64_t op_idx = 0;
    for (auto it = std::next(loopBodyBlock->begin(), beginIdx);
         it != std::next(srcBlockEnd); it++) {
      Operation* clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++) {
      lastYielded[i] = operandMap.lookup(yieldedValues[i]);
    }
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = std::next(loopBodyBlock->begin(), beginIdx);
       it != std::next(srcBlockEnd); it++) {
    annotateFn(0, &*it, builder);
  }

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);

  // Try to interleave instructions.
  // Note we do not support to interleave ops with nested blocks.
  for (auto& op : *loopBodyBlock) {
    if (op.getNumRegions() != 0) {
      return success();
    }
  }

  // Check that the number of cloned instructions is correct.
  int64_t numEffectiveOps = std::distance(loopBodyBlock->begin(),
                                          std::prev(loopBodyBlock->end(), 1)) -
                            beginIdx;
  if (numEffectiveOps != numOps * unrollFactor) {
    return failure();
  }

  // Make sure `toMove` is after `toMoveAfter` when calling this lambda
  // function.
  auto canMoveAfter = [&](Block::iterator toMove, Block::iterator toMoveAfter) {
    if (toMove->getBlock() != toMoveAfter->getBlock() ||
        !toMoveAfter->isBeforeInBlock(&(*toMove))) {
      return false;
    }
    bool canMove = true;
    for (auto iter = std::prev(toMove); iter != toMoveAfter;
         std::advance(iter, -1)) {
      if (dependsOnInBlock(&(*toMove), &(*iter))) {
        canMove = false;
        break;
      }
    }
    return canMove;
  };

  bool canMove = true;
  // Note only the first `numOps - 1` ops need to be moved to do interleaving.
  for (int64_t opIdx = 0; opIdx < (numOps - 1) && canMove; opIdx++) {
    auto firstInst =
        std::next(loopBodyBlock->begin(), beginIdx + opIdx * unrollFactor);
    auto toMoveAfter = firstInst;
    for (int64_t unrollIdx = 1; unrollIdx < unrollFactor && canMove;
         unrollIdx++) {
      auto toInterleave = std::next(firstInst, (numOps - opIdx) * unrollIdx);
      canMove &= canMoveAfter(toInterleave, toMoveAfter);
      if (canMove) {
        toInterleave->moveAfter(&(*toMoveAfter));
        std::advance(toMoveAfter, 1);
      }
    }
  }

  return success();
}

// This function unrolls the given for-loop op and interleaves the instructions
LogicalResult loopUnrollByFactorAndTryInterleave(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations())) {
    return success();
  }

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

    if (unrollFactor == 1) {
      if (tripCount == 1 && failed(promoteIfSingleIteration(forOp))) {
        return failure();
      }
      return success();
    }

    int64_t tripCountEvenMultiple = tripCount - (tripCount % unrollFactor);
    int64_t upperBoundUnrolledCst = lbCst + tripCountEvenMultiple * stepCst;
    assert(upperBoundUnrolledCst <= ubCst);
    int64_t stepUnrolledCst = stepCst * unrollFactor;

    // Create constant for 'upperBoundUnrolled' and set epilogue loop flag.
    generateEpilogueLoop = upperBoundUnrolledCst < ubCst;
    if (generateEpilogueLoop) {
      upperBoundUnrolled = boundsBuilder.create<arith::ConstantIndexOp>(
          loc, upperBoundUnrolledCst);
    } else {
      upperBoundUnrolled = ubCstOp;
    }

    // Create constant for 'stepUnrolled'.
    stepUnrolled =
        stepCst == stepUnrolledCst
            ? step
            : boundsBuilder.create<arith::ConstantIndexOp>(loc, stepUnrolledCst)
                  .getResult();
  } else {
    // Dynamic loop bounds computation.
    // TODO: Add dynamic asserts for negative lb/ub/step, or
    // consider using ceilDiv from AffineApplyExpander.
    auto lowerBound = forOp.getLowerBound();
    auto upperBound = forOp.getUpperBound();
    Value diff =
        boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value tripCount = boundsBuilder.create<arith::CeilDivUIOp>(loc, diff, step);
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

  generateUnrolledLoopMayInterleave(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv + step * i;
        auto stride = b.create<arith::MulIOp>(
            loc, step, b.create<arith::ConstantIndexOp>(loc, i));
        return b.create<arith::AddIOp>(loc, iv, stride);
      },
      annotateFn, iterArgs, yieldedValues);
  // Promote the loop body up if this has turned into a single iteration loop.
  (void)promoteIfSingleIteration(forOp);
  return success();
}

void createAlignMemrefWithTile(OpBuilder& b, Value memref, int64_t tile_size) {
  assert(memref != nullptr);
  auto memref_ty = memref.getType().cast<MemRefType>();
  int byte_width = memref_ty.getElementTypeBitWidth() / 8;
  if (byte_width == 0) {
    // Currently, load and store are at least aligned to 1 byte.
    byte_width = 1;
  }
  Location loc = memref.getLoc();
  b.create<memref::AssumeAlignmentOp>(loc, memref, byte_width * tile_size);
}

}  // namespace disc_ral
}  // namespace mlir
