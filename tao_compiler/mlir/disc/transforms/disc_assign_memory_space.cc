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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"   // TF:llvm-project
#include "mlir/IR/Location.h"     // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h"
#include "mlir/disc/transforms/placement_utils.h"

// This file implements logic to assign memory space for each memref type.

// Currently we only support:
//  - SCF, not CFG
//  - single function (a.k.a. entry function), not cross function

namespace mlir {
namespace disc_ral {

namespace {

using placement_utils::copyWithMemorySpace;

// return a new memref type with provided memory space if the input type if a
// memref type otherwise return the original type.
Type maybeConvert(MLIRContext* ctx, Type type, StringRef memory_space) {
  if (auto memref = type.dyn_cast<MemRefType>()) {
    return copyWithMemorySpace(ctx, memref, memory_space);
  }
  return type;
}

// Returns true if the type is memref.
bool isMemRefType(Type t) { return t.dyn_cast<MemRefType>() != nullptr; }

LogicalResult updateAssignment(DenseMap<Value, StringRef>& assignment,
                               Value value, StringRef memory_space,
                               bool& converged) {
  auto it = assignment.find(value);
  if (it == assignment.end()) {
    converged = false;
    assignment[value] = memory_space;
    return success();
  }
  return (it->second == memory_space) ? success() : failure();
}

LogicalResult mergeAssignment(DenseMap<Value, StringRef>& assignment, Value lhs,
                              Value rhs, bool& converged) {
  auto lhs_it = assignment.find(lhs);
  auto rhs_it = assignment.find(rhs);

  if (lhs_it == assignment.end() && rhs_it == assignment.end()) {
    return success();
  }

  if (lhs_it != assignment.end() && rhs_it != assignment.end()) {
    return (lhs_it->second == rhs_it->second) ? success() : failure();
  }

  converged = false;
  if (lhs_it != assignment.end()) {
    assignment[rhs] = lhs_it->second;
  } else {
    assignment[lhs] = rhs_it->second;
  }
  return success();
}

struct DiscAssignMemorySpacePass
    : public DiscAssignMemorySpacePassBase<DiscAssignMemorySpacePass> {
  DiscAssignMemorySpacePass(const std::string& entry_func_name,
                            bool gpu_enabled)
      : DiscAssignMemorySpacePassBase<
            DiscAssignMemorySpacePass>::DiscAssignMemorySpacePassBase() {
    this->entry_func_name_ = entry_func_name;
    this->gpu_enabled_ = gpu_enabled;
  }

  LogicalResult processLmhloOperation(
      Operation* op, const SmallVectorImpl<StringRef>& input_placements,
      const SmallVectorImpl<StringRef>& output_placements,
      DenseMap<Value, StringRef>& assignment, bool& converged);

  LogicalResult processOperation(
      Operation* op, const SmallVectorImpl<StringRef>& input_placements,
      const SmallVectorImpl<StringRef>& output_placements,
      DenseMap<Value, StringRef>& assignment, bool& converged);

  LogicalResult processBlock(
      Block* block, const SmallVectorImpl<StringRef>& input_placements,
      const SmallVectorImpl<StringRef>& output_placements,
      DenseMap<Value, StringRef>& assignment, bool& converged);

  LogicalResult processRegion(
      Region* region, const SmallVectorImpl<StringRef>& input_placements,
      const SmallVectorImpl<StringRef>& output_placements,
      DenseMap<Value, StringRef>& assignment, bool& converged);

  LogicalResult applyAssignment(
      func::FuncOp main, DenseMap<Value, StringRef>& assignment,
      const SmallVectorImpl<StringRef>& input_placements,
      const SmallVectorImpl<StringRef>& output_placements);

  LogicalResult applyRegionAssignment(Region*,
                                      DenseMap<Value, StringRef>& assignment);

  LogicalResult applyBlockAssignment(Block*,
                                     DenseMap<Value, StringRef>& assignment);

  LogicalResult applyOperationAssignment(
      Operation*, DenseMap<Value, StringRef>& assignment);

  LogicalResult cloneSmallLmhloConstOps(ModuleOp m);

  void runOnOperation() override {
    ModuleOp m = getOperation();
    func::FuncOp main = m.lookupSymbol<func::FuncOp>(entry_func_name_);
    if (!main) {
      m.emitError("entry func: " + entry_func_name_ + " not found");
      signalPassFailure();
      return;
    }

    if (failed(cloneSmallLmhloConstOps(m))) {
      m.emitError("failed to clone small lmhlo const ops");
      signalPassFailure();
      return;
    }

    // init input/output placements of the entry function
    SmallVector<StringRef, 4> input_placements;
    SmallVector<StringRef, 4> output_placements;
    if (failed(placement_utils::parseEntryFunctionInputPlacements(
            main, gpu_enabled_, input_placements)) ||
        failed(placement_utils::parseEntryFunctionOutputPlacements(
            main, gpu_enabled_, output_placements))) {
      main.emitError("failed to parse i/o placements");
      signalPassFailure();
      return;
    }

    // mapping a value to its placement info
    DenseMap<Value, StringRef> assignment;

    // Try to propagate memory space assignment.
    bool converged;
    do {
      converged = true;
      if (failed(processRegion(&main.getBody(), input_placements,
                               output_placements, assignment, converged))) {
        main.emitError("failed to propagate placement info");
        signalPassFailure();
        return;
      }
    } while (!converged);

    // Apply memory space assignment
    if (failed(applyAssignment(main, assignment, input_placements,
                               output_placements))) {
      main.emitError("failed to apply assignment");
      signalPassFailure();
    }
  }
};

// Our placer in mhlo world will assign a placement for each const op. However
// such placement will be dropped out during const folding process (e.g.
// canonicalize pass). Furthermore,  CSE pass may fuse two const op having same
// value while different placements into one const op, leading to invalid IR.
//  Example is:
//  ```
//    %0 = memref.alloc() : memref<i32>
//    "lmhlo.constant"(%0)...
//    "lmhlo.reduce"(%arg0, %0) {...} // suppose this reduce is placed on cpu
//    "lmhlo.reduce"(%arg1, %0) {...} // suppose this reduce is placed on gpu
//    // %0 will have cross-device consumer no matter what placement it has.
//  ```
// This is a workaround for the above problem. We clone small const values to
// make sure that each lmhlo const only have one consumer. This solution have
// some limitations. One major problem is that we suppose that large const will
// not be shared by lmhlo consumers placed on different devices.
LogicalResult DiscAssignMemorySpacePass::cloneSmallLmhloConstOps(ModuleOp m) {
  SmallVector<lmhlo::ConstantOp> constOps;
  m.walk([&](lmhlo::ConstantOp constOp) {
    Value out = constOp->getOperand(0);
    if (constOp->getParentOfType<lmhlo::ReduceOp>() ||
        constOp->getParentOfType<lmhlo::FusionOp>()) {
      return;
    }
    if (IsSmallBuffer(out)) {
      constOps.push_back(constOp);
    }
  });

  for (lmhlo::ConstantOp constOp : constOps) {
    // Use set vector to make sure having deterministic iteration order.
    SetVector<Operation*> lmhloUsers;
    Value out = constOp->getOperand(0);
    for (Operation* user : out.getUsers()) {
      if (user == constOp.getOperation()) continue;
      if (isa<lmhlo::LmhloOp>(user)) lmhloUsers.insert(user);
    }
    if (lmhloUsers.size() < 2) continue;
    // clone the const for the first (n-1) users, let the last user use the
    // original const op.
    lmhloUsers.pop_back();
    constOp->removeAttr(placement_utils::kDiscPlaceAssignment);
    for (Operation* user : lmhloUsers) {
      OpBuilder b(user);
      Location loc = user->getLoc();
      Value newOut =
          b.create<memref::AllocOp>(loc, out.getType().cast<MemRefType>());
      Operation* clonedConstOp = b.clone(*constOp.getOperation());
      clonedConstOp->replaceUsesOfWith(out, newOut);
      user->replaceUsesOfWith(out, newOut);
    }
  }
  return success();
}

LogicalResult DiscAssignMemorySpacePass::processRegion(
    Region* region, const SmallVectorImpl<StringRef>& input_placements,
    const SmallVectorImpl<StringRef>& output_placements,
    DenseMap<Value, StringRef>& assignment, bool& converged) {
  // Only SCF is supported a.t.m.
  if (region->getBlocks().size() != 1) {
    getOperation().emitError("suppose a single block inside the region");
    return failure();
  }
  return processBlock(&region->getBlocks().front(), input_placements,
                      output_placements, assignment, converged);
}

LogicalResult DiscAssignMemorySpacePass::processBlock(
    Block* block, const SmallVectorImpl<StringRef>& input_placements,
    const SmallVectorImpl<StringRef>& output_placements,
    DenseMap<Value, StringRef>& assignment, bool& converged) {
  // mapping block arguments
  for (auto&& z : llvm::zip(block->getArguments(), input_placements)) {
    if (failed(updateAssignment(assignment, std::get<0>(z), std::get<1>(z),
                                converged))) {
      return failure();
    }
  }

  // mapping returnOp's operands
  Operation& returnOp = block->back();
  assert(returnOp.hasTrait<OpTrait::ReturnLike>());
  for (auto&& z : llvm::zip(returnOp.getOperands(), output_placements)) {
    if (failed(updateAssignment(assignment, std::get<0>(z), std::get<1>(z),
                                converged))) {
      return failure();
    }
  }

  for (Operation& op : block->getOperations()) {
    // skip the last return-like ops since we have processed it above.
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      continue;
    }

    if (failed(processOperation(&op, input_placements, output_placements,
                                assignment, converged))) {
      return op.emitOpError() << "fail to propagate placement info for op";
    }
  }

  return success();
}

LogicalResult DiscAssignMemorySpacePass::processOperation(
    Operation* op, const SmallVectorImpl<StringRef>& input_placements,
    const SmallVectorImpl<StringRef>& output_placements,
    DenseMap<Value, StringRef>& assignment, bool& converged) {
  // Skip these metadata only ops since they don't change placement.
  if (isa<shape::ShapeOfOp, memref::DimOp>(op)) {
    return success();
  }

  // memref view/cast ops
  // result & operand memref should have same assignment
  if (isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
          memref::ReinterpretCastOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getResult(0);
    return mergeAssignment(assignment, operand, result, converged);
  }

  // host to device copy
  // input should be placed on cpu and output should be placed on gpu.
  if (isa<lmhlo_disc::H2DOp>(op)) {
    LogicalResult status = updateAssignment(assignment, op->getOperand(0),
                                            placement_utils::kCpu, converged);
    if (!failed(status)) {
      status = updateAssignment(assignment, op->getOperand(1),
                                placement_utils::kGpu, converged);
    }
    return status;
  }

  // device to host copy
  // input should be placed on gpu and output should be placed on cpu.
  if (isa<lmhlo_disc::D2HOp>(op)) {
    LogicalResult status = updateAssignment(assignment, op->getOperand(0),
                                            placement_utils::kGpu, converged);
    if (!failed(status)) {
      status = updateAssignment(assignment, op->getOperand(1),
                                placement_utils::kCpu, converged);
    }
    return status;
  }

  if (auto customCallV2Op = dyn_cast<lmhlo_disc::CustomCallV2Op>(op)) {
    auto parseAndApply = [&](StringRef s, ValueRange vs) {
      SmallVector<StringRef, 4> parsedItems;
      s.split(parsedItems, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      if (parsedItems.size() != vs.size()) return failure();
      for (const auto& z : llvm::zip(parsedItems, vs)) {
        auto placement = std::get<0>(z);
        if (placement == "s" || placement == "h" ||
            placement == "x" && !this->gpu_enabled_) {
          if (failed(updateAssignment(assignment, std::get<1>(z),
                                      placement_utils::kCpu, converged)))
            return failure();
        } else if (placement == "d" || placement == "x" && this->gpu_enabled_) {
          if (failed(updateAssignment(assignment, std::get<1>(z),
                                      placement_utils::kGpu, converged)))
            return failure();
        } else {
          return failure();
        }
      }
      return success();
    };
    if (failed(parseAndApply(customCallV2Op.getInputPlacements(),
                             op->getOperands())))
      return op->emitError()
             << " failed to parse and apply the input placements\n";
    if (failed(parseAndApply(customCallV2Op.getOutputPlacements(),
                             op->getResults())))
      return op->emitError()
             << " failed to parse and apply the output placements\n";
    return success();
  }

  // process lmhlo ops
  if (isa<lmhlo::LmhloOp>(op)) {
    return processLmhloOperation(op, input_placements, output_placements,
                                 assignment, converged);
  }

  // TODO(disc): support scf ops.
  if (op->getNumRegions() != 0) {
    return op->emitOpError() << "scf is not supported a.t.m.";
  }

  // Skip if no memref types involved.
  if (llvm::none_of(op->getOperandTypes(), isMemRefType) &&
      llvm::none_of(op->getResultTypes(), isMemRefType)) {
    return success();
  }

  // TODO(disc): using an unsupported op list
  // TODO(disc): support cross function inference.
  if (isa<func::CallOp>(op)) {
    return op->emitOpError() << "function call is not supported a.t.m.";
  }

  // memref that is explicitly operated by load/store ops are supposed to be
  // placed on cpu.
  if (isa<memref::LoadOp>(op)) {
    return updateAssignment(assignment, op->getOperand(0),
                            placement_utils::kCpu, converged);
  }
  if (isa<memref::StoreOp>(op)) {
    return updateAssignment(assignment, op->getOperand(1),
                            placement_utils::kCpu, converged);
  }

  // TODO(disc): check other ops.

  return success();
}

LogicalResult DiscAssignMemorySpacePass::processLmhloOperation(
    Operation* op, const SmallVectorImpl<StringRef>& input_placements,
    const SmallVectorImpl<StringRef>& output_placements,
    DenseMap<Value, StringRef>& assignment, bool& converged) {
  // Some operands of the lmhlo ops have to be placed on cpu.
  auto shape_operands = placement_utils::getShapeCalcOperandList(op);
  for (int idx : shape_operands) {
    if (failed(updateAssignment(assignment, op->getOperand(idx),
                                placement_utils::kCpu, converged))) {
      return failure();
    }
  }

  // non-shape operands
  SmallVector<Value, 4> non_shape_operands;
  for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (llvm::find(shape_operands, i) != shape_operands.end()) continue;
    non_shape_operands.push_back(op->getOperand(i));
  }

  // All non shape operands are supposed to have same placements. Try to verify
  // and propagate placement info between operands.
  StringRef placement;
  auto attr =
      op->getAttrOfType<StringAttr>(placement_utils::kDiscPlaceAssignment);
  if (attr) {
    placement = attr.cast<StringAttr>().getValue();
  }

  for (Value non_shape_operand : non_shape_operands) {
    auto it = assignment.find(non_shape_operand);
    if (it == assignment.end()) continue;
    if (!placement.empty() && it->second != placement) {
      return op->emitError()
             << "non shape operands not have same placements " << placement
             << " vs " << it->second << " (expected vs actual)\n";
    }
    placement = it->second;
  }

  if (placement.empty()) {
    // No explicit placement.
    return success();
  }

  LogicalResult status = success();
  for (Value not_shape_operand : non_shape_operands) {
    if (placement == placement_utils::kCpu) {
      status = updateAssignment(assignment, not_shape_operand,
                                placement_utils::kCpu, converged);
    } else if (placement == placement_utils::kGpu) {
      status = updateAssignment(assignment, not_shape_operand,
                                placement_utils::kGpu, converged);
    } else {
      // unknown placement
      status = failure();
    }
    if (failed(status)) return status;
  }

  return success();
}

LogicalResult DiscAssignMemorySpacePass::applyAssignment(
    func::FuncOp main, DenseMap<Value, StringRef>& assignment,
    const SmallVectorImpl<StringRef>& input_placements,
    const SmallVectorImpl<StringRef>& output_placements) {
  // apply assignment inside the region
  if (failed(applyRegionAssignment(&main.getBody(), assignment)))
    return failure();

  // update entry function type.
  MLIRContext* ctx = main.getContext();
  SmallVector<Type, 4> input_types;
  SmallVector<Type, 4> output_types;
  llvm::transform(
      llvm::zip(main.getFunctionType().getInputs(), input_placements),
      std::back_inserter(input_types),
      [&](const std::tuple<Type, StringRef>& v) {
        return maybeConvert(ctx, std::get<0>(v), std::get<1>(v));
      });
  llvm::transform(
      llvm::zip(main.getFunctionType().getResults(), output_placements),
      std::back_inserter(output_types),
      [&](const std::tuple<Type, StringRef>& v) {
        return maybeConvert(ctx, std::get<0>(v), std::get<1>(v));
      });

  // Update entry function type.
  main.setType(FunctionType::get(ctx, input_types, output_types));
  return success();
}

LogicalResult DiscAssignMemorySpacePass::applyRegionAssignment(
    Region* region, DenseMap<Value, StringRef>& assignment) {
  auto main = cast<func::FuncOp>(region->getParentOp());
  for (Block& block : llvm::make_early_inc_range(*region)) {
    if (failed(applyBlockAssignment(&block, assignment))) return failure();
  }
  return success();
}

LogicalResult DiscAssignMemorySpacePass::applyBlockAssignment(
    Block* block, DenseMap<Value, StringRef>& assignment) {
  // update block argument
  int originalArgNum = block->getNumArguments();
  for (int i = 0; i < originalArgNum; ++i) {
    // always fetch the first argument.
    Value arg = block->getArgument(0);
    Type ty = arg.getType();
    auto it = assignment.find(arg);
    if (it != assignment.end()) {
      ty = maybeConvert(arg.getContext(), ty, it->second);
    }
    Value newArg = block->addArgument(ty, arg.getLoc());
    arg.replaceAllUsesWith(newArg);
    if (it != assignment.end()) {
      assignment[newArg] = it->second;
      assignment.erase(arg);
    }
    // remove the old block argument
    block->eraseArgument(0);
  }

  auto main = cast<func::FuncOp>(block->getParentOp());
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (failed(applyOperationAssignment(&op, assignment))) return failure();
  }
  return success();
}

template <typename OpTy>
Operation* replaceResultType(Operation* op,
                             DenseMap<Value, StringRef>& assignment) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  SmallVector<Type> newResultTypes;
  for (Value oldValue : op->getResults()) {
    Type oldType = oldValue.getType();
    auto it = assignment.find(oldValue);
    if (it != assignment.end()) {
      newResultTypes.push_back(
          maybeConvert(op->getContext(), oldType, it->second));
    } else {
      newResultTypes.push_back(oldType);
    }
  }
  auto newOp =
      b.create<OpTy>(loc, newResultTypes, op->getOperands(), op->getAttrs());
  for (const auto& z : llvm::zip(op->getResults(), newOp->getResults())) {
    Value oldValue = std::get<0>(z);
    Value newValue = std::get<1>(z);
    oldValue.replaceAllUsesWith(newValue);
    auto it = assignment.find(oldValue);
    if (it != assignment.end()) {
      assignment[newValue] = it->second;
      assignment.erase(oldValue);
    }
  }
  op->erase();
  return newOp;
}

template <typename OpTy>
Operation* tryReplaceResultType(Operation* op,
                                DenseMap<Value, StringRef>& assignment) {
  if (!isa<OpTy>(op)) return nullptr;
  return replaceResultType<OpTy>(op, assignment);
}

template <typename First, typename Second, typename... OtherOpList>
Operation* tryReplaceResultType(Operation* op,
                                DenseMap<Value, StringRef>& assignment) {
  if (Operation* newOp = tryReplaceResultType<First>(op, assignment))
    return newOp;
  return tryReplaceResultType<Second, OtherOpList...>(op, assignment);
}

LogicalResult DiscAssignMemorySpacePass::applyOperationAssignment(
    Operation* op, DenseMap<Value, StringRef>& assignment) {
  if (llvm::none_of(op->getResults(),
                    [&](Value v) { return assignment.count(v) != 0; })) {
    return success();
  }

  // clang-format: off
  Operation* newOp = tryReplaceResultType<
      memref::AllocOp, memref::AllocaOp, memref::SubViewOp, memref::ViewOp,
      memref::CastOp, memref::ReinterpretCastOp, lmhlo_disc::CustomCallV2Op>(
      op, assignment);
  // clang-format: on

  if (newOp) {
    return success();
  }

  return op->emitOpError()
         << "failed to replace the result type of an unsupported op";
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscAssignMemorySpacePass(
    const std::string& entry_func_name, bool gpu_enabled) {
  return std::make_unique<DiscAssignMemorySpacePass>(entry_func_name,
                                                     gpu_enabled);
}

}  // namespace disc_ral
}  // namespace mlir
