// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/MLIRContext.h"     // TF:llvm-project
#include "mlir/Pass/Pass.h"          // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {

using placement_utils::kCpu;
using placement_utils::kDiscShapeCalcAttr;
using placement_utils::kGpu;
using placement_utils::PlacementType;

namespace disc_ral {
namespace {

// This pass explicitly marks the shape calculating Op by adding an Attr. Nested
// FuncOps should be taken into consideration.
// Following Ops are shape Ops:
//  - Shape Op's operands
//  - Shape operands according to kShapeCalcOperandMap
//  - GetDimensionSizeOp
struct DiscMarkShapeCalc
    : public DiscMarkShapeCalculationPassBase<DiscMarkShapeCalc> {
 public:
  using DiscMarkShapeCalculationPassBase<
      DiscMarkShapeCalc>::DiscMarkShapeCalculationPassBase;

  void runOnOperation() override;

 private:
  // Mark shape calculation subgraph
  void MarkShapeCalcOps();

  // Update marked set.
  // Add some operands of dynamic shape OPs into marked set according to lookup
  // table.
  void markShapeCalculationOps(func::FuncOp func,
                               DenseSet<Operation*>& marked_ops);

  // Update marked set.
  // If a OP is in marked set, add all of its operands to marked set.
  void inferOperands(func::FuncOp func, llvm::DenseSet<Operation*>& marked_ops);
};

void DiscMarkShapeCalc::runOnOperation() {
  // Mark shape calculation subgraph
  MarkShapeCalcOps();
};

// Mark the Ops that is the producer of any shape operands
// TODO(disc): handle when TupleOp exists in shape_calc_ops
void DiscMarkShapeCalc::MarkShapeCalcOps() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  llvm::DenseSet<Operation*> shape_calc_ops;

  mlir::func::FuncOp func = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!func) return signalPassFailure();

  markShapeCalculationOps(func, shape_calc_ops);

  inferOperands(func, shape_calc_ops);

  for (Operation* op : shape_calc_ops) {
    // We suppose that mhlo op only has single output, either having tensor
    // type or tuple type.
    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      SmallVector<Attribute, 4> attrs(tp.size(), builder.getBoolAttr(true));
      op->setAttr(kDiscShapeCalcAttr, ArrayAttr::get(tp.getContext(), attrs));
    } else {
      op->setAttr(kDiscShapeCalcAttr, builder.getBoolAttr(true));
    }
  }
}

// NOTE(lanbo.llb): mhlo.disc op will produce 2 outputs, the 2nd output is i64
// tensor with single element represent number of valid elements in the 1st
// output, this will be used to calc the final output shape. It is indeed a
// shape calc, but where op should not be considered as shape calc op. By
// marking it as shape calc op will cause mhlo_disc.where impossible to fuse in
// later fusion pass.
bool withInputFromWhereOp(Operation* op) {
  for (Value operand : op->getOperands()) {
    auto op = operand.getDefiningOp();
    if (op != nullptr && isa<mhlo_disc::WhereOp>(op)) {
      return true;
    }
  }
  return false;
}

void DiscMarkShapeCalc::markShapeCalculationOps(
    func::FuncOp func, llvm::DenseSet<Operation*>& marked_ops) {
  auto& block = func.getBlocks().front();
  for (Operation& op : block) {
    // TODO(disc): If the operand of the op is a nested FuncOp, mark the
    // associated producer in the nested FuncOp
    if (!placement_utils::isMarkShapeCalcTargetOp(&op)) continue;
    if (!marked_ops.contains(&op)) {
      // Mark following Ops into shape calculation set
      if (isa<mhlo::GetDimensionSizeOp, tensor::FromElementsOp,
              tensor::ExtractOp>(&op) &&
          !withInputFromWhereOp(&op)) {
        marked_ops.insert(&op);
        continue;
      }

      // Mark operands into shape calculation set according to the lookup table.
      const auto& shape_operand_indices =
          placement_utils::getShapeCalcOperandList(&op);

      for (auto operand_idx : shape_operand_indices) {
        auto operand = op.getOperand(operand_idx).getDefiningOp();
        if (operand == nullptr) continue;
        if (!placement_utils::isMarkShapeCalcTargetOp(operand)) {
          continue;
        }
        marked_ops.insert(operand);
      }
    }
  };
}

void DiscMarkShapeCalc::inferOperands(func::FuncOp func,
                                      llvm::DenseSet<Operation*>& marked_ops) {
  auto& block = func.getBlocks().front();
  for (auto& op : llvm::make_early_inc_range(
           llvm::make_range(block.rbegin(), block.rend()))) {
    if (!placement_utils::isMarkShapeCalcTargetOp(&op)) {
      continue;
    }
    // If the op is already in shape calculation op set, insert all of its
    // operands into shape calculation op set
    if (marked_ops.contains(&op)) {
      for (auto operand_value : op.getOperands()) {
        Operation* operand = operand_value.getDefiningOp();
        if (operand == nullptr) continue;
        if (!placement_utils::isMarkShapeCalcTargetOp(operand)) {
          continue;
        }
        if (isa<tensor::DimOp>(operand) || isa<shape::ShapeOfOp>(operand)) {
          continue;
        }
        // large tensors are more likely to be 'data' computations, which are
        // supposed to be placed on device if possible.
        // TODO(disc): add a rule for dynamic shape case as well.
        //   Possible solutions:
        //   - using rank information as a hint (higher rank tensors are more
        //   likely to be data computation)
        //   - using likely value after we refactor shape constriant
        //   implementation.
        auto ty = operand_value.getType().dyn_cast<RankedTensorType>();
        if (ty && ty.hasStaticShape() && ty.getNumElements() > 64) {
          continue;
        }
        if (!withInputFromWhereOp(&op)) {
          marked_ops.insert(operand);
        }
      }
    }
  };
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscMarkShapeCalcOpPass() {
  return std::make_unique<DiscMarkShapeCalc>();
}

}  // namespace disc_ral
}  // namespace mlir
