// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/utils/source_emitter.h"

namespace mlir {
namespace disc_ral {

////////////////////// Dot GPU FusionStrategy Implemenation ////////////
////////////////////////////////////////////////////////////////////////

bool DotGpuFusionStrategy::isFusible(Operation* op) {
  // Only scalar const are supported by the fusion codegen engine a.t.m.
  if (isa<lmhlo::ConstantOp>(op)) {
    auto constant = cast<lmhlo::ConstantOp>(op);
    MemRefType type = constant.getOutput().getType().cast<MemRefType>();
    return (type.getRank() == 0 || constant.getValue().isSplat());
  }

  return isa<lmhlo::DotGeneralOp>(op) || SourceEmitterCUDA::isSupportedOp(op);
}

bool DotGpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                             FusionPattern& fusion_pattern) {
  /*
   * 1. Only one dot is supported per fusion, currently.
   * 2. Dot is the first op;
   * 3. Other ops are supported by the CUDA source emitter;
   * 4. Other ops' effective operands are not the operands of the fusion,
   *    currently; otherwise this op should be scalar/splat constant.
   *
   * The conditions 1 and 4 will be loosed in the future.
   */

  const auto& op_list = fusion_pattern.getOpList();
  const auto& operands = fusion_pattern.getOperands();
  const auto& results = fusion_pattern.getResults();

  SmallVector<Operation*> dot_ops;
  SmallVector<Operation*> mem_intensive_ops;
  for (auto op : op_list) {
    if (isa<lmhlo::DotGeneralOp>(op)) {
      dot_ops.push_back(op);
    } else {
      mem_intensive_ops.push_back(op);
    }
  }

  // If it is a single supported elementwise op, it may be fused with dot later.
  // Do not do multi-op fusion for pure element-wise ops to prevent breaking
  // possible dot fusion.
  if (dot_ops.size() == 0 && mem_intensive_ops.size() == 1) {
    return initFusionPatternBase(shapeAnalysis, fusion_pattern);
  }

  // Only one dot.
  if (dot_ops.size() != 1) {
    return false;
  }

  // All the effective-operand of non-dot ops are not the operand of the fusion.
  DenseSet<Value> operand_set(operands.begin(), operands.end());
  for (auto op : mem_intensive_ops) {
    SmallVector<Value> effective_operands = getEffectiveOperands(op);
    for (auto in : effective_operands) {
      if (operand_set.contains(in)) {
        return false;
      }
    }
  }

  fusion_pattern.setFusionType(FusionType::kDot);
  fusion_pattern.setDominantOp(dot_ops[0]);

  return true;
}

bool DotGpuFusionStrategy::finalizeFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fused_pattern,
    SmallVectorImpl<Operation*>& excluded_ops) {
  return true;
}

bool DotGpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                   FusionPattern& lhs, FusionPattern& rhs,
                                   FusionPattern& target) {
  auto& operands = target.getOperands();
  auto& results = target.getResults();

  if (results.size() + operands.size() >
      options_.max_num_arguments_per_kernel) {
    // some backend devices (e.g. GPU) do not support a kernel with
    // too many arguments.
    return false;
  }

  // TODO: ConstantOp is allowed to be fused as the output op temporarily, which
  // will be moved out in a later pass.

  LLVM_DEBUG(llvm::dbgs() << "DotGpuFusionStrategy::tryFuse success()\n");
  return true;
}

SmallVector<Value> DotGpuFusionStrategy::getEffectiveOperands(Operation* op) {
  SmallVector<Value> effective_operands;
  if (isa<lmhlo::DynamicBroadcastInDimOp, lmhlo::DynamicReshapeOp,
          lmhlo::TransposeOp>(op)) {
    effective_operands.push_back(op->getOperand(0));
  } else {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    effective_operands = op->getOperands().take_front(num_input_operand);
  }
  return effective_operands;
}

}  // namespace disc_ral
}  // namespace mlir