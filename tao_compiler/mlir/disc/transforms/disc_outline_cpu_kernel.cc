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

// This file implements the logic to outline each cpu kernel (represented as a
// parallelOp) to a dedicated function.

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_ral_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

/// Identifies operations that are beneficial to sink into kernels. These
/// operations may not have side-effects, as otherwise sinking (and hence
/// duplicating them) is not legal.
static bool isSinkingBeneficiary(Operation* op) {
  return isa<arith::ConstantOp, memref::DimOp, arith::SelectOp, arith::CmpIOp>(
      op);
}

/// For a given operation `op`, computes whether it is beneficial to sink the
/// operation into the kernel. An operation can be sunk if doing so does not
/// introduce new kernel arguments. Whether a value is already available in the
/// kernel (and hence does not introduce new arguments) is checked by
/// querying `existingDependencies` and `availableValues`.
/// If an operand is not yet available, we recursively check whether it can be
/// made available by siking its defining op.
/// Operations that are indentified for sinking are added to `beneficiaryOps` in
/// the order they should appear in the kernel. Furthermore, `availableValues`
/// is updated with results that will be available after sinking the identified
/// ops.
static bool extractBeneficiaryOps(
    Operation* op, SetVector<Value> existingDependencies,
    SetVector<Operation*>& beneficiaryOps,
    llvm::SmallPtrSetImpl<Value>& availableValues) {
  if (beneficiaryOps.count(op)) return true;

  if (!isSinkingBeneficiary(op)) return false;

  for (Value operand : op->getOperands()) {
    // It is already visible in the kernel, keep going.
    if (availableValues.count(operand)) continue;
    // Else check whether it can be made available via sinking or already is a
    // dependency.
    Operation* definingOp = operand.getDefiningOp();
    if ((!definingOp ||
         !extractBeneficiaryOps(definingOp, existingDependencies,
                                beneficiaryOps, availableValues)) &&
        !existingDependencies.count(operand))
      return false;
  }
  // We will sink the operation, mark its results as now available.
  beneficiaryOps.insert(op);
  for (Value result : op->getResults()) availableValues.insert(result);
  return true;
}

LogicalResult sinkOperationsIntoLaunchOp(Region& launchOpBody) {
  // Identify uses from values defined outside of the scope of the launch
  // operation.
  SetVector<Value> sinkCandidates;
  getUsedValuesDefinedAbove(launchOpBody, sinkCandidates);

  SetVector<Operation*> toBeSunk;
  llvm::SmallPtrSet<Value, 4> availableValues;
  for (Value operand : sinkCandidates) {
    Operation* operandOp = operand.getDefiningOp();
    if (!operandOp) continue;
    extractBeneficiaryOps(operandOp, sinkCandidates, toBeSunk, availableValues);
  }

  // Insert operations so that the defs get cloned before uses.
  BlockAndValueMapping map;
  OpBuilder builder(launchOpBody);
  for (Operation* op : toBeSunk) {
    Operation* clonedOp = builder.clone(*op, map);
    // Only replace uses within the launch op.
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 launchOpBody);
  }
  return success();
}

/// Outline the `launch` operation body into a kernel function. Replace
/// `scf.yield` operations by `return` in the generated function.
static func::FuncOp outlineKernelFuncImpl(Operation* launchOp,
                                          StringRef kernelFnName,
                                          SetVector<Value>& operands) {
  Location loc = launchOp->getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp->getContext());
  Region& launchOpBody = launchOp->getRegion(0);

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(launchOpBody, operands);

  // Create the gpu.func operation.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    kernelOperandTypes.push_back(operand.getType());
  }
  FunctionType type =
      FunctionType::get(launchOp->getContext(), kernelOperandTypes, {});
  auto outlinedFunc = builder.create<func::FuncOp>(loc, kernelFnName, type);
  outlinedFunc->setAttr(kCpuKernelFunc, builder.getUnitAttr());

  BlockAndValueMapping map;

  // Map arguments from launch region to the arguments of the func
  // operation.
  Block& entryBlock = *outlinedFunc.addEntryBlock();
  Region& outlinedFuncBody = outlinedFunc.getBody();
  for (auto operand : enumerate(operands))
    map.map(operand.value(), entryBlock.getArgument(operand.index()));

  // Clone the region of the launch operation into the func operation.
  // TODO: If cloneInto can be modified such that if a mapping for
  // a block exists, that block will be used to clone operations into (at the
  // end of the block), instead of creating a new block, this would be much
  // cleaner.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Branch from entry of the func operation to the block that is cloned
  // from the entry block of the gpu.launch operation.
  Block& launchOpEntry = launchOpBody.front();
  Block* clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<cf::BranchOp>(loc, clonedLaunchOpEntry);

  SetVector<Operation*> toBeRemoved;
  outlinedFunc.walk([&](scf::YieldOp op) {
    if (op->getParentOfType<scf::IfOp>() ||
        op->getParentOfType<scf::ParallelOp>() ||
        op->getParentOfType<scf::ForOp>()) {
      return;
    }
    toBeRemoved.insert(op);
  });
  for (Operation* op : toBeRemoved) {
    OpBuilder replacer(op);
    assert(op->getNumResults() == 0);
    replacer.create<func::ReturnOp>(op->getLoc());
    op->erase();
  }
  return outlinedFunc;
}

func::FuncOp outlineKernelFunc(Operation* launchOp, StringRef kernelFnName,
                               llvm::SmallVectorImpl<Value>& operands) {
  DenseSet<Value> inputOperandSet;
  inputOperandSet.insert(operands.begin(), operands.end());
  SetVector<Value> operandSet(operands.begin(), operands.end());
  auto funcOp = outlineKernelFuncImpl(launchOp, kernelFnName, operandSet);
  for (auto operand : operandSet) {
    if (!inputOperandSet.count(operand)) operands.push_back(operand);
  }
  return funcOp;
}

std::string getKernelName(int index, scf::ParallelOp op, StringRef funcName) {
  lmhlo::FusionOp fusion = op->getParentOfType<lmhlo::FusionOp>();
  auto strIndex =
      (index != 0 ? ("_" + llvm::Twine(index)) : llvm::Twine("")).str();
  if (fusion) {
    // fusionName already contains funcName
    auto fusionName = getFusionFullName(fusion);
    return (llvm::Twine(fusionName) + strIndex).str();
  }
  return (funcName + llvm::Twine("_gKernel_") + strIndex).str();
};

/// Replace `parallel` operations with an `disc_ral.cpu_launch` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `parallelOp` with constant region arguments inlined.
static void convertToLaunchFuncOp(Operation* launchOp, func::FuncOp kernelFunc,
                                  ValueRange operands) {
  Location loc = launchOp->getLoc();
  OpBuilder builder(launchOp);
  // TODO(disc): find a way to estimate the unit workload size.
  Value unitWorkloadSizeHint = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<disc_ral::CpuLaunchOp>(
      loc, operands[0], operands[1], operands[2], operands[3],
      unitWorkloadSizeHint, operands.drop_front(4),
      mlir::SymbolRefAttr::get(kernelFunc));
  launchOp->erase();
}

LogicalResult cloneAndMoveTo(OpBuilder& b, scf::ParallelOp parallelOp,
                             ValueRange lower, ValueRange upper,
                             ValueRange step, Block* block) {
  auto cloned = cast<scf::ParallelOp>(b.clone(*parallelOp.getOperation()));
  cloned->setOperands(0, lower.size(), lower);
  cloned->setOperands(lower.size(), upper.size(), upper);
  cloned->setOperands(lower.size() + upper.size(), step.size(), step);
  cloned->moveBefore(block, block->begin());

  // Move out memref.alloc/dealloc op from the loop op.
  // They are used to manage the tile buffers. We can re-use these buffers cross
  // iterations in the same thread.
  SmallVector<Operation*> allocOps;
  SmallVector<Operation*> deallocOps;
  for (Operation& op : cloned.getLoopBody().front()) {
    if (isa<memref::AllocOp>(&op)) {
      allocOps.push_back(&op);
    } else if (isa<memref::DeallocOp>(&op)) {
      deallocOps.push_back(&op);
    }
  }
  for (Operation* op : allocOps) {
    op->moveBefore(cloned);
  }
  for (Operation* op : llvm::reverse(deallocOps)) {
    op->moveAfter(cloned);
  }

  if (cloned
          .walk([&](LoopLikeOpInterface loopLike) {
            moveLoopInvariantCode(loopLike);
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  moveLoopInvariantCode(cloned);
  return success();
}

LogicalResult rewriteLaunchOpSetting(scf::ParallelOp parallelOp,
                                     Operation*& targetOp,
                                     SmallVector<Value>& operands) {
  Location loc = parallelOp.getLoc();
  OpBuilder builder(parallelOp);
  int numIvs = parallelOp.getInductionVars().size();
  auto launchSettingType = MemRefType::get(
      {numIvs}, builder.getIndexType(), MemRefLayoutAttrInterface(),
      StringAttr::get(parallelOp->getContext(), placement_utils::kCpu));
  Value lowerBound = builder.create<memref::AllocaOp>(loc, launchSettingType);
  Value upperBound = builder.create<memref::AllocaOp>(loc, launchSettingType);
  Value step = builder.create<memref::AllocaOp>(loc, launchSettingType);
  for (auto&& en : llvm::enumerate(llvm::zip(parallelOp.getLowerBound(),
                                             parallelOp.getUpperBound(),
                                             parallelOp.getStep()))) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, en.index());
    builder.create<memref::StoreOp>(loc, std::get<0>(en.value()), lowerBound,
                                    idx);
    builder.create<memref::StoreOp>(loc, std::get<1>(en.value()), upperBound,
                                    idx);
    builder.create<memref::StoreOp>(loc, std::get<2>(en.value()), step, idx);
  }
  operands.push_back(lowerBound);
  operands.push_back(upperBound);
  operands.push_back(step);

  // We use a scf::if op as the wrapper op. The pred of the if op is a constant
  // true.
  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, llvm::None, pred, false);
  Block* thenBlock = &ifOp.getThenRegion().getBlocks().front();
  parallelOp->moveBefore(thenBlock, thenBlock->begin());
  targetOp = ifOp;

  // Rewrite the launch setting of the parallel op.
  // We also add a speculation logic for the steps of the loop since only loops
  // with step equal to one will be applied auto vectorization in LLVM.
  builder.setInsertionPoint(parallelOp);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value stepAllOnes = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  SmallVector<Value> lowerBoundVec, upperBoundVec, stepVec,
      stepOneVec(numIvs, one);
  for (int i = 0; i < numIvs; ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    lowerBoundVec.push_back(
        builder.create<memref::LoadOp>(loc, lowerBound, idx));
    upperBoundVec.push_back(
        builder.create<memref::LoadOp>(loc, upperBound, idx));
    stepVec.push_back(builder.create<memref::LoadOp>(loc, step, idx));
    Value stepIsOne = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, stepVec.back(), one);
    stepAllOnes = builder.create<arith::AndIOp>(loc, stepAllOnes, stepIsOne);
  }
  ifOp = builder.create<scf::IfOp>(loc, llvm::None, stepAllOnes, true);
  if (failed(cloneAndMoveTo(builder, parallelOp, lowerBoundVec, upperBoundVec,
                            stepOneVec,
                            &ifOp.getThenRegion().getBlocks().front()))) {
    return ifOp->emitError("failed to cloned the parallel op\n");
  }
  if (failed(cloneAndMoveTo(builder, parallelOp, lowerBoundVec, upperBoundVec,
                            stepVec,
                            &ifOp.getElseRegion().getBlocks().front()))) {
    return ifOp->emitError("failed to cloned the parallel op\n");
  }
  parallelOp->erase();
  return success();
}

struct DiscOutlineCpuKernel : DiscOutlineCpuKernelBase<DiscOutlineCpuKernel> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<func::FuncOp> candidateFuncOps;
    for (func::FuncOp func : m.getOps<func::FuncOp>()) {
      if (func->getAttrOfType<UnitAttr>(kCpuKernelFunc)) continue;
      candidateFuncOps.push_back(func);
    }
    if (candidateFuncOps.empty()) return;

    SymbolTable symbolTable(getOperation());
    for (func::FuncOp func : candidateFuncOps) {
      if (failed(processFunction(symbolTable, func))) {
        signalPassFailure();
        return;
      }
    }
  }

  LogicalResult processFunction(SymbolTable& symbolTable, func::FuncOp func);
};

LogicalResult DiscOutlineCpuKernel::processFunction(SymbolTable& symbolTable,
                                                    func::FuncOp func) {
  SmallVector<scf::ParallelOp> parallelOps;
  func.walk([&](scf::ParallelOp op) {
    // skip nested parallel op.
    if (op->getParentOfType<scf::ParallelOp>()) return;
    // skip small cpu kernel.
    if (op->getAttrOfType<UnitAttr>(kSmallCpuKernel)) return;
    // TODO(disc): skip gpu parallel op.
    // TODO(disc): figure out a way to distinguish cpu and gpu parallel op.
    parallelOps.push_back(op);
  });

  // Insert just after the function.
  Block::iterator insertPt(func->getNextNode());
  for (auto&& en : llvm::enumerate(parallelOps)) {
    // operands for the outlined function.
    SmallVector<Value> operands;
    operands.push_back(func.getArgument(0));

    auto kernelFnName = getKernelName(en.index(), en.value(), func.getName());

    // Create launch setting operands (e.g. lower/upper/step) and rewrite
    // parallelOp accordingly. To wrap the targeting parallelOp, in order to
    // make sure that all new inserted ops are in the same region. The wrapper
    // op itself doesn't matter, except that it providing a region
    // representation.
    Operation* wrapperOp = nullptr;
    if (failed(rewriteLaunchOpSetting(en.value(), wrapperOp, operands))) {
      return en.value().emitError("failed to rewriteLaunchOpSetting");
    }
    assert(wrapperOp != nullptr);

    LLVM_DEBUG(llvm::dbgs() << "wrapperOp = " << *wrapperOp << "\n");

    // Pull in instructions that can be sunk
    if (failed(sinkOperationsIntoLaunchOp(wrapperOp->getRegion(0)))) {
      return wrapperOp->emitError("failed to sink operations into parallel op");
    }

    func::FuncOp outlinedFunc =
        outlineKernelFunc(wrapperOp, kernelFnName, operands);
    symbolTable.insert(outlinedFunc, insertPt);
    convertToLaunchFuncOp(wrapperOp, outlinedFunc, operands);
  }
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscOutlineCpuKernelPass() {
  return std::make_unique<DiscOutlineCpuKernel>();
}

}  // namespace disc_ral
}  // namespace mlir
