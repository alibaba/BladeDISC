// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// #include "iree-dialects/Dialect/LinalgTransform/SimpleRewriterBase.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"
#include "mlir/disc/tools/disc-transform/utils.h"

//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace disc_ral {
namespace transform_dialect {

using namespace mlir;
using namespace mlir::scf;

namespace {
/// Helper to keep internal information during pipelining transformation.
struct LoopPipelinerInternal {
  /// Coarse liverange information for ops used across stages.
  struct LiverangeInfo {
    unsigned lastUseStage = 0;
    unsigned defStage = 0;
  };

 protected:
  ForOp forOp;
  unsigned maxStage = 0;
  DenseMap<Operation*, unsigned> stages;
  std::vector<Operation*> opOrder;
  int64_t ub;
  int64_t lb;
  int64_t step;
  PipeliningOption::AnnotationlFnType annotateFn = nullptr;
  bool peelEpilogue;
  PipeliningOption::PredicateOpFn predicateFn = nullptr;

  // When peeling the kernel we generate several version of each value for
  // different stage of the prologue. This map tracks the mapping between
  // original Values in the loop and the different versions
  // peeled from the loop.
  DenseMap<Value, llvm::SmallVector<Value>> valueMapping;

  /// Assign a value to `valueMapping`, this means `val` represents the version
  /// `idx` of `key` in the epilogue.
  void setValueMapping(Value key, Value el, int64_t idx);

 public:
  /// Initalize the information for the given `op`, return true if it
  /// satisfies the pre-condition to apply pipelining.
  bool initializeLoopInfo(ForOp op, const PipeliningOption& options);
  /// Emits the prologue, this creates `maxStage - 1` part which will contain
  /// operations from stages [0; i], where i is the part index.
  void emitPrologue(RewriterBase& rewriter);
  /// Gather liverange information for Values that are used in a different stage
  /// than its definition.
  llvm::MapVector<Value, LiverangeInfo> analyzeCrossStageValues();
  scf::ForOp createKernelLoop(
      const llvm::MapVector<Value, LiverangeInfo>& crossStageValues,
      RewriterBase& rewriter,
      llvm::DenseMap<std::pair<Value, unsigned>, unsigned>& loopArgMap);
  /// Emits the pipelined kernel. This clones loop operations following user
  /// order and remaps operands defined in a different stage as their use.
  void createKernel(
      scf::ForOp newForOp,
      const llvm::MapVector<Value, LiverangeInfo>& crossStageValues,
      const llvm::DenseMap<std::pair<Value, unsigned>, unsigned>& loopArgMap,
      RewriterBase& rewriter);
  /// Emits the epilogue, this creates `maxStage - 1` part which will contain
  /// operations from stages [i; maxStage], where i is the part index.
  llvm::SmallVector<Value> emitEpilogue(RewriterBase& rewriter);
};

bool LoopPipelinerInternal::initializeLoopInfo(
    ForOp op, const PipeliningOption& options) {
  forOp = op;
  // TODO: add support for dynamic shape of K
  auto upperBoundCst =
      forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lowerBoundCst =
      forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBoundCst || !lowerBoundCst) return false;
  ub = upperBoundCst.value();
  lb = lowerBoundCst.value();
  auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!stepCst) return false;
  step = stepCst.value();
  peelEpilogue = options.peelEpilogue;
  predicateFn = options.predicateFn;
  if (!peelEpilogue && predicateFn == nullptr) return false;
  int64_t numIteration = ceilDiv(ub - lb, step);
  std::vector<std::pair<Operation*, unsigned>> schedule;
  options.getScheduleFn(forOp, schedule);
  if (schedule.empty()) return false;

  opOrder.reserve(schedule.size());
  for (auto& opSchedule : schedule) {
    maxStage = std::max(maxStage, opSchedule.second);
    stages[opSchedule.first] = opSchedule.second;
    opOrder.push_back(opSchedule.first);
  }
  if (numIteration <= maxStage) return false;

  // All operations need to have a stage.
  for (Operation& op : forOp.getBody()->without_terminator()) {
    if (stages.find(&op) == stages.end()) {
      op.emitOpError("not assigned a pipeline stage");
      return false;
    }
  }

  // Currently, we do not support assigning stages to ops in nested regions. The
  // block of all operations assigned a stage should be the single `scf.for`
  // body block.
  for (const auto& [op, stageNum] : stages) {
    (void)stageNum;
    if (op == forOp.getBody()->getTerminator()) {
      op->emitError("terminator should not be assigned a stage");
      return false;
    }
    if (op->getBlock() != forOp.getBody()) {
      op->emitOpError(
          "the owning Block of all operations assigned a stage "
          "should be the loop body block");
      return false;
    }
  }

  // Only support loop carried dependency with a distance of 1. This means the
  // source of all the scf.yield operands needs to be defined by operations in
  // the loop.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [this](Value operand) {
                     Operation* def = operand.getDefiningOp();
                     return !def || stages.find(def) == stages.end();
                   }))
    return false;
  annotateFn = options.annotateFn;
  return true;
}

/// Clone `op` and call `callback` on the cloned op's oeprands as well as any
/// operands of nested ops that:
/// 1) aren't defined within the new op or
/// 2) are block arguments.
static Operation* cloneAndUpdateOperands(
    RewriterBase& rewriter, Operation* op,
    function_ref<void(OpOperand* newOperand)> callback) {
  Operation* clone = rewriter.clone(*op);
  for (OpOperand& operand : clone->getOpOperands()) callback(&operand);
  clone->walk([&](Operation* nested) {
    for (OpOperand& operand : nested->getOpOperands()) {
      Operation* def = operand.get().getDefiningOp();
      if ((def && !clone->isAncestor(def)) ||
          operand.get().isa<BlockArgument>())
        callback(&operand);
    }
  });
  return clone;
}

void LoopPipelinerInternal::emitPrologue(RewriterBase& rewriter) {
  // Initialize the iteration argument to the loop initiale values.
  for (BlockArgument& arg : forOp.getRegionIterArgs()) {
    OpOperand& operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (int64_t i = 0; i < maxStage; i++) {
    // special handling for induction variable as the increment is implicit.
    Value iv =
        rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), lb + i * step);
    setValueMapping(forOp.getInductionVar(), iv, i);
    for (Operation* op : opOrder) {
      if (stages[op] > i) continue;
      Operation* newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand* newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[i - stages[op]];
              newOperand->set(replacement);
            }
          });
      if (annotateFn)
        annotateFn(newOp, PipeliningOption::PipelinerPart::Prologue, i);
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        i - stages[op]);
        // If the value is a loop carried dependency update the loop argument
        // mapping.
        for (OpOperand& operand : yield->getOpOperands()) {
          if (operand.get() != op->getResult(destId)) continue;
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],
                          newOp->getResult(destId), i - stages[op] + 1);
        }
      }
    }
  }
}

llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
LoopPipelinerInternal::analyzeCrossStageValues() {
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo> crossStageValues;
  for (Operation* op : opOrder) {
    unsigned stage = stages[op];

    auto analyzeOperand = [&](OpOperand& operand) {
      Operation* def = operand.get().getDefiningOp();
      if (!def) return;
      auto defStage = stages.find(def);
      if (defStage == stages.end() || defStage->second == stage) return;
      assert(stage > defStage->second);
      LiverangeInfo& info = crossStageValues[operand.get()];
      info.defStage = defStage->second;
      info.lastUseStage = std::max(info.lastUseStage, stage);
    };

    for (OpOperand& operand : op->getOpOperands()) analyzeOperand(operand);
    visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
      analyzeOperand(*operand);
    });
  }
  return crossStageValues;
}

scf::ForOp LoopPipelinerInternal::createKernelLoop(
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>&
        crossStageValues,
    RewriterBase& rewriter,
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned>& loopArgMap) {
  // Creates the list of initial values associated to values used across
  // stages. The initial values come from the prologue created above.
  // Keep track of the kernel argument associated to each version of the
  // values passed to the kernel.
  llvm::SmallVector<Value> newLoopArg;
  // For existing loop argument initialize them with the right version from the
  // prologue.
  for (const auto& retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation* def = retVal.value().getDefiningOp();
    assert(def && "Only support loop carried dependencies of distance 1");
    unsigned defStage = stages[def];
    Value valueVersion = valueMapping[forOp.getRegionIterArgs()[retVal.index()]]
                                     [maxStage - defStage];
    assert(valueVersion);
    newLoopArg.push_back(valueVersion);
  }
  for (auto escape : crossStageValues) {
    LiverangeInfo& info = escape.second;
    Value value = escape.first;
    for (unsigned stageIdx = 0; stageIdx < info.lastUseStage - info.defStage;
         stageIdx++) {
      Value valueVersion =
          valueMapping[value][maxStage - info.lastUseStage + stageIdx];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
      loopArgMap[std::make_pair(value, info.lastUseStage - info.defStage -
                                           stageIdx)] = newLoopArg.size() - 1;
    }
  }

  // Create the new kernel loop. When we peel the epilgue we need to peel
  // `numStages - 1` iterations. Then we adjust the upper bound to remove those
  // iterations.
  Value newUb = forOp.getUpperBound();
  if (peelEpilogue)
    newUb = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(),
                                                    ub - maxStage * step);
  auto newForOp =
      rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), newUb,
                                  forOp.getStep(), newLoopArg);
  // When there are no iter args, the loop body terminator will be created.
  // Since we always create it below, remove the terminator if it was created.
  if (!newForOp.getBody()->empty())
    rewriter.eraseOp(newForOp.getBody()->getTerminator());
  return newForOp;
}

/// Replace any use of `target` with `replacement` in `op`'s operands or within
/// `op`'s nested regions.
static void replaceInOp(Operation* op, Value target, Value replacement) {
  for (auto& use : llvm::make_early_inc_range(target.getUses())) {
    if (op->isAncestor(use.getOwner())) use.set(replacement);
  }
}

/// Given a cloned op in the new kernel body, updates induction variable uses.
/// We replace it with a version incremented based on the stage where it is
/// used.
static void updateInductionVariableUses(RewriterBase& rewriter, Location loc,
                                        Operation* newOp, Value newForIv,
                                        unsigned maxStage, unsigned useStage,
                                        unsigned step) {
  rewriter.setInsertionPoint(newOp);
  Value offset = rewriter.create<arith::ConstantIndexOp>(
      loc, (maxStage - useStage) * step);
  Value iv = rewriter.create<arith::AddIOp>(loc, newForIv, offset);
  replaceInOp(newOp, newForIv, iv);
  rewriter.setInsertionPointAfter(newOp);
}

/// If the value is a loop carried value coming from stage N + 1 remap, it will
/// become a direct use.
static void updateIterArgUses(RewriterBase& rewriter, IRMapping& bvm,
                              Operation* newOp, ForOp oldForOp, ForOp newForOp,
                              unsigned useStage,
                              const DenseMap<Operation*, unsigned>& stages) {
  for (unsigned i = 0; i < oldForOp.getNumRegionIterArgs(); i++) {
    Value yieldedVal = oldForOp.getBody()->getTerminator()->getOperand(i);
    Operation* dep = yieldedVal.getDefiningOp();
    if (!dep) continue;
    auto stageDep = stages.find(dep);
    if (stageDep == stages.end() || stageDep->second == useStage) continue;
    if (stageDep->second != useStage + 1) continue;
    Value replacement = bvm.lookup(yieldedVal);
    replaceInOp(newOp, newForOp.getRegionIterArg(i), replacement);
  }
}

/// For operands defined in a previous stage we need to remap it to use the
/// correct region argument. We look for the right version of the Value based
/// on the stage where it is used.
static void updateCrossStageUses(
    RewriterBase& rewriter, Operation* newOp, IRMapping& bvm, ForOp newForOp,
    unsigned useStage, const DenseMap<Operation*, unsigned>& stages,
    const llvm::DenseMap<std::pair<Value, unsigned>, unsigned>& loopArgMap) {
  // Because we automatically cloned the sub-regions, there's no simple way
  // to walk the nested regions in pairs of (oldOps, newOps), so we just
  // traverse the set of remapped loop arguments, filter which ones are
  // relevant, and replace any uses.
  for (auto [remapPair, newIterIdx] : loopArgMap) {
    auto [crossArgValue, stageIdx] = remapPair;
    Operation* def = crossArgValue.getDefiningOp();
    assert(def);
    unsigned stageDef = stages.lookup(def);
    if (useStage <= stageDef || useStage - stageDef != stageIdx) continue;

    // Use "lookupOrDefault" for the target value because some operations
    // are remapped, while in other cases the original will be present.
    Value target = bvm.lookupOrDefault(crossArgValue);
    Value replacement = newForOp.getRegionIterArg(newIterIdx);

    // Replace uses in the new op's operands and any nested uses.
    replaceInOp(newOp, target, replacement);
  }
}

void LoopPipelinerInternal::createKernel(
    scf::ForOp newForOp,
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>&
        crossStageValues,
    const llvm::DenseMap<std::pair<Value, unsigned>, unsigned>& loopArgMap,
    RewriterBase& rewriter) {
  valueMapping.clear();

  // Create the kernel, we clone instruction based on the order given by
  // user and remap operands coming from a previous stages.
  rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (const auto& arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  }
  SmallVector<Value> predicates(maxStage + 1, nullptr);
  if (!peelEpilogue) {
    // Create a predicate for each stage except the last stage.
    for (unsigned i = 0; i < maxStage; i++) {
      Value c = rewriter.create<arith::ConstantIndexOp>(
          newForOp.getLoc(), ub - (maxStage - i) * step);
      Value pred = rewriter.create<arith::CmpIOp>(
          newForOp.getLoc(), arith::CmpIPredicate::slt,
          newForOp.getInductionVar(), c);
      predicates[i] = pred;
    }
  }
  for (Operation* op : opOrder) {
    int64_t useStage = stages[op];
    auto* newOp = rewriter.clone(*op, mapping);
    // Within the kernel body, update uses of the induction variable, uses of
    // the original iter args, and uses of cross stage values.
    updateInductionVariableUses(rewriter, forOp.getLoc(), newOp,
                                newForOp.getInductionVar(), maxStage,
                                stages[op], step);
    updateIterArgUses(rewriter, mapping, newOp, forOp, newForOp, useStage,
                      stages);
    updateCrossStageUses(rewriter, newOp, mapping, newForOp, useStage, stages,
                         loopArgMap);

    if (predicates[useStage]) {
      newOp = predicateFn(rewriter, newOp, predicates[useStage]);
      // Remap the results to the new predicated one.
      for (auto values : llvm::zip(op->getResults(), newOp->getResults()))
        mapping.map(std::get<0>(values), std::get<1>(values));
    }
    rewriter.setInsertionPointAfter(newOp);
    if (annotateFn)
      annotateFn(newOp, PipeliningOption::PipelinerPart::Kernel, 0);
  }

  // Collect the Values that need to be returned by the forOp. For each
  // value we need to have `LastUseStage - DefStage` number of versions
  // returned.
  // We create a mapping between original values and the associated loop
  // returned values that will be needed by the epilogue.
  llvm::SmallVector<Value> yieldOperands;
  for (Value retVal : forOp.getBody()->getTerminator()->getOperands()) {
    yieldOperands.push_back(mapping.lookupOrDefault(retVal));
  }
  for (auto& it : crossStageValues) {
    int64_t version = maxStage - it.second.lastUseStage + 1;
    unsigned numVersionReturned = it.second.lastUseStage - it.second.defStage;
    // add the original verstion to yield ops.
    // If there is a liverange spanning across more than 2 stages we need to add
    // extra arg.
    for (unsigned i = 1; i < numVersionReturned; i++) {
      setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                      version++);
      yieldOperands.push_back(
          newForOp.getBody()->getArguments()[yieldOperands.size() + 1 +
                                             newForOp.getNumInductionVars()]);
    }
    setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                    version++);
    yieldOperands.push_back(mapping.lookupOrDefault(it.first));
  }
  // Map the yield operand to the forOp returned value.
  for (const auto& retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation* def = retVal.value().getDefiningOp();
    assert(def && "Only support loop carried dependencies of distance 1");
    unsigned defStage = stages[def];
    setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                    newForOp->getResult(retVal.index()),
                    maxStage - defStage + 1);
  }
  rewriter.create<scf::YieldOp>(forOp.getLoc(), yieldOperands);
}

llvm::SmallVector<Value> LoopPipelinerInternal::emitEpilogue(
    RewriterBase& rewriter) {
  llvm::SmallVector<Value> returnValues(forOp->getNumResults());
  // Emit different versions of the induction variable. They will be
  // removed by dead code if not used.
  for (int64_t i = 0; i < maxStage; i++) {
    Value newlastIter = rewriter.create<arith::ConstantIndexOp>(
        forOp.getLoc(), lb + step * ((((ub - 1) - lb) / step) - i));
    setValueMapping(forOp.getInductionVar(), newlastIter, maxStage - i);
  }
  // Emit `maxStage - 1` epilogue part that includes operations from stages
  // [i; maxStage].
  for (int64_t i = 1; i <= maxStage; i++) {
    for (Operation* op : opOrder) {
      if (stages[op] < i) continue;
      Operation* newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand* newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[maxStage - stages[op] + i];
              newOperand->set(replacement);
            }
          });
      if (annotateFn)
        annotateFn(newOp, PipeliningOption::PipelinerPart::Epilogue, i - 1);
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        maxStage - stages[op] + i);
        // If the value is a loop carried dependency update the loop argument
        // mapping and keep track of the last version to replace the original
        // forOp uses.
        for (OpOperand& operand :
             forOp.getBody()->getTerminator()->getOpOperands()) {
          if (operand.get() != op->getResult(destId)) continue;
          unsigned version = maxStage - stages[op] + i + 1;
          // If the version is greater than maxStage it means it maps to the
          // original forOp returned value.
          if (version > maxStage) {
            returnValues[operand.getOperandNumber()] = newOp->getResult(destId);
            continue;
          }
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],
                          newOp->getResult(destId), version);
        }
      }
    }
  }
  return returnValues;
}

void LoopPipelinerInternal::setValueMapping(Value key, Value el, int64_t idx) {
  auto it = valueMapping.find(key);
  // If the value is not in the map yet add a vector big enough to store all
  // versions.
  if (it == valueMapping.end())
    it =
        valueMapping
            .insert(std::make_pair(key, llvm::SmallVector<Value>(maxStage + 1)))
            .first;
  it->second[idx] = el;
}

}  // namespace

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";
static const StringLiteral kPipeliningExtraBarrier =
    "__pipelining_extra_barrier__";

/// Returns a new predicated operation to support unpeeled epilogue. Unpeeled
/// epilogue needs to handle the last iterations within the mainloop which
/// requires predicating operations, for e.g., OOB global memory access. This
/// helper function predicates operations (where predication is avialable),
/// checks if unpredicated operations are side-effect free and acceptable to
/// execute speculatively.
static Operation* replaceOpWithPredicatedOp(RewriterBase& rewriter,
                                            Operation* op, Value pred) {
  // Predication is only supported for AsyncCopyOp. Thus, for operations which
  // are *not* AsyncCopyOp additional checks are requrired in order to be issued
  // speculatively.
  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) {
    // Return/execute the op if it is a side effect free.
    if (mlir::isMemoryEffectFree(op)) return op;
    // Return/execute the op if it is barrier, commit group, or ldmatrix op.
    if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp, nvgpu::LdMatrixOp,
            nvgpu::DeviceAsyncWaitOp>(op))
      return op;
    // Return/execute the op if it is a shared memory load.
    if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      auto loadBaseType = llvm::cast<MemRefType>(loadOp.getBase().getType());
      if (hasSharedMemoryAddressSpace(loadBaseType)) return op;
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      auto loadBaseType = loadOp.getMemRefType();
      if (hasSharedMemoryAddressSpace(loadBaseType)) return op;
    }
    // If we are here that means the operation does not have predication support
    // and cannot be speculatively executed. Thus, unpeeled epilogue is not
    // supported.
    assert(false &&
           "Unpeeled epilogue not supported with a side-effect instruction "
           "with no predication.");
  }

  // Replace mainloop AsyncCopy with AsyncCopy(zfill) inline asm.
  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  auto loc = asyncCopyOp->getLoc();

  // Create srcElement Value based on the pred.
  // The next few lins generate the below code:
  // srcElement = (pred) ?  prevSrcElements : 0;
  Value dstElements =
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());
  Value originalSrcElement =
      asyncCopyOp.getSrcElements() ? asyncCopyOp.getSrcElements() : dstElements;
  Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto srcElements =
      rewriter.create<arith::SelectOp>(loc, pred, originalSrcElement, c0Index);
  int64_t sizeInBytes =
      (asyncCopyOp.getDst().getType().getElementTypeBitWidth() *
       asyncCopyOp.getDstElements().getZExtValue()) /
      8;
  UnitAttr bypassL1 = sizeInBytes == 16 ? rewriter.getUnitAttr() : UnitAttr();
  auto asyncCopyZfillOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(asyncCopyOp.getContext()),
      asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), asyncCopyOp.getSrc(),
      asyncCopyOp.getSrcIndices(), asyncCopyOp.getDstElements(), srcElements,
      bypassL1);
  rewriter.eraseOp(asyncCopyOp);

  // Return the newly create predicated AsyncCopyZfillOp.
  return asyncCopyZfillOp;
}

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation*>& dep, Operation* op,
                      Block* block) {
  if (!dep.insert(op).second) return;
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block) addDepOps(dep, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic by default, put load from global
/// memory in stage 0 and the rest in stage 1. If store_stage = 0 then put store
/// to shared memory in stage 0 as well.
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of stage 0 ops.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningFirstStage)) {
      addDepOps(loadDep, &op, forOp.getBody());
    }
  }
  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage `maxDepth`. In order to have a correct scheduling even with back
  // edges we order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, depth));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  }
}

static void setAsyncAnnotations(Operation* op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth,
                                PipeliningSchedulingStrategy schedule) {
  if (auto waitOp = dyn_cast<NVVM::CpAsyncWaitGroupOp>(op)) {
    // Based on the order copies within the loop we need to adjust the number of
    // copies in flight.
    bool copyBeforeLoad =
        schedule == PipeliningSchedulingStrategy::nvidiaTensorCore;
    if (waitOp.getN()) return;
    int numGroupInFlight = 0;
    if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
        part == scf::PipeliningOption::PipelinerPart::Prologue) {
      numGroupInFlight = copyBeforeLoad ? depth - 2 : depth - 1;
    } else {
      // By construction there should be no wait op in the prologue as all the
      // wait should be in the last stage.
      assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
      // Based on the schedule we pick we know how many groups are in flight for
      // each iteration of the epilogue.
      numGroupInFlight = depth - 1 - iteration;
    }
    OpBuilder b(op);
    waitOp.setNAttr(b.getI32IntegerAttr(numGroupInFlight));
  } else if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
    unsigned pipelineStoreStage =
        schedule == PipeliningSchedulingStrategy::loadStoreStage0 ? 0 : 1;
    if (pipelineStoreStage != 0 ||
        part != mlir::scf::PipeliningOption::PipelinerPart::Prologue ||
        iteration >= depth - 1)
      return;
    OpBuilder b(op);
    barrierOp->setAttr(kPipeliningExtraBarrier, b.getUnitAttr());
  }
}

/// Check if the for operations contains a shared memory copy that can be
/// pipelined and annotate operations with stage information if this is the
/// case.
static bool setPipeliningMarkers(scf::ForOp forOp, bool pipelineStoreStage) {
  bool copyToWorkgroupMemory = false;
  OpBuilder builder(forOp.getContext());
  SmallVector<Operation*> barriers;
  for (Operation& op : forOp.getBody()->getOperations()) {
    // do not check region number since
    // we have mma loops inside cta-k-loop
    // clang-off
    // Pipeline the most inner for op that should be a flat region.
    // if (op.getNumRegions() > 0)
    //   return false;
    // clang-on
    if (isa<gpu::BarrierOp>(op)) {
      barriers.push_back(&op);
      if (pipelineStoreStage == 0)
        op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    }
    if (isa<nvgpu::DeviceAsyncCopyOp, NVVM::CpAsyncCommitGroupOp>(op)) {
      copyToWorkgroupMemory = true;
      op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      // async copy ops need to be moved along with previous barrier.
      for (Operation* barrier : barriers) {
        barrier->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      }
      barriers.clear();
      continue;
    }
    auto ld = dyn_cast<vector::TransferReadOp>(op);
    if (!ld) continue;
    auto ldSrcType = llvm::cast<MemRefType>(ld.getSource().getType());
    if (!ld->hasOneUse()) continue;
    auto st = dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
    if (!st) continue;
    auto stSrcType = llvm::cast<MemRefType>(st.getSource().getType());
    if (!hasSharedMemoryAddressSpace(stSrcType)) continue;
    copyToWorkgroupMemory = true;
    ld->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    if (pipelineStoreStage == 0)
      st->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
  }
  if (copyToWorkgroupMemory) {
    forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
    if (pipelineStoreStage == 0 && !barriers.empty()) {
      barriers.front()->erase();
    }
  }
  return copyToWorkgroupMemory;
}

// Apply pipeline rewrite pattern assuming the operations were already
// annotated with stage information.
// TODO: move away from using attribute annotations.
FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, int64_t depth,
                                      bool epiloguePeeling,
                                      PipeliningSchedulingStrategy schedule) {
  // TODO: Refactor schedules to not rely on markers.
  if (schedule == PipeliningSchedulingStrategy::loadGlobalStage0 ||
      schedule == PipeliningSchedulingStrategy::loadStoreStage0) {
    unsigned pipelineStoreStage =
        schedule == PipeliningSchedulingStrategy::loadGlobalStage0;
    if (!setPipeliningMarkers(forOp, pipelineStoreStage)) {
      return failure();
    }
  }

  scf::PipeliningOption options;
  unsigned maxDepth = depth;
  auto getSchedule = [maxDepth, schedule](
                         scf::ForOp forOp,
                         std::vector<std::pair<Operation*, unsigned>>& ops) {
    return getPipelineStages(forOp, ops, maxDepth);
  };
  auto setAnnotation = [maxDepth, schedule](
                           Operation* op,
                           scf::PipeliningOption::PipelinerPart part,
                           unsigned iteration) {
    return setAsyncAnnotations(op, part, iteration, maxDepth, schedule);
  };
  options.getScheduleFn = getSchedule;
  options.annotateFn = setAnnotation;

  // Use un-peeled epilogue (i.e. epiloguePeeling=flase) only when predication
  // is avialable a.k.a. AsyncCopyOp.
  if (!epiloguePeeling) {
    options.peelEpilogue = false;
    options.predicateFn = [](RewriterBase& rewriter, Operation* op,
                             Value pred) {
      return replaceOpWithPredicatedOp(rewriter, op, pred);
    };
  }
  LoopPipelinerInternal pipeliner;
  if (!pipeliner.initializeLoopInfo(forOp, options)) {
    return failure();
  }
  IRRewriter rewriter(forOp->getContext());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  // 1. Emit prologue.
  pipeliner.emitPrologue(rewriter);

  // 2. Track values used across stages. When a value cross stages it will
  // need to be passed as loop iteration arguments.
  // We first collect the values that are used in a different stage than where
  // they are defined.
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
      crossStageValues = pipeliner.analyzeCrossStageValues();

  // Mapping between original loop values used cross stage and the block
  // arguments associated after pipelining. A Value may map to several
  // arguments if its liverange spans across more than 2 stages.
  llvm::DenseMap<std::pair<Value, unsigned>, unsigned> loopArgMap;
  // 3. Create the new kernel loop and return the block arguments mapping.
  ForOp newForOp =
      pipeliner.createKernelLoop(crossStageValues, rewriter, loopArgMap);
  // Create the kernel block, order ops based on user choice and remap
  // operands.
  pipeliner.createKernel(newForOp, crossStageValues, loopArgMap, rewriter);
  llvm::SmallVector<Value> returnValues =
      newForOp.getResults().take_front(forOp->getNumResults());
  if (options.peelEpilogue) {
    // 4. Emit the epilogue after the new forOp.
    rewriter.setInsertionPointAfter(newForOp);
    returnValues = pipeliner.emitEpilogue(rewriter);
  }
  // 5. Erase the original loop and replace the uses with the epilogue output.
  if (forOp->getNumResults() > 0)
    rewriter.replaceOp(forOp, returnValues);
  else
    rewriter.eraseOp(forOp);
  return newForOp;
}

}  // namespace transform_dialect
}  // namespace disc_ral
}  // namespace mlir
