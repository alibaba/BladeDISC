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

bool isOpFusible(Operation* op) {
  // Only scalar const are supported by the fusion codegen engine a.t.m.
  if (isa<lmhlo::ConstantOp>(op)) {
    auto constant = cast<lmhlo::ConstantOp>(op);
    MemRefType type = constant.getOutput().getType().cast<MemRefType>();
    return (type.getRank() == 0 || constant.getValue().isSplat());
  }

  return SourceEmitterCUDA::isSupportedOp(op);
}

bool PreDotGpuFusionStrategy::isFusible(Operation* op) {
  return isOpFusible(op);
}

Value PreDotGpuFusionStrategy::getEffectiveShape(FusionPattern& target,
                                                 Value v) {
  return v;
}

bool DotGpuFusionStrategy::isFusible(Operation* op) {
  return isa<lmhlo::DotGeneralOp>(op) || isOpFusible(op);
}

bool DotGpuFusionStrategy::isFusible(FusionPattern& fusion_pattern) {
  for (Operation* op : fusion_pattern.getOpList()) {
    if (!isFusible(op)) {
      return false;
    }
  }
  return true;
}

bool DotGpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                             FusionPattern& fusion_pattern) {
#if 0
  llvm::errs() << "[ZZ] the fusion pattern is:\n";
  dumpFusionPattern(fusion_pattern);
#endif
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

{
  void getDirectDependentOps(Operation * op,
                             DenseSet<Operation*> & dependences) {
    dependences.clear();
    SmallVector<Value> affectedValues;
    for (auto result : op->getResults()) {
      affectedValues.push_back(result);
    }
    for (auto operand : op->getOperands()) {
      if (IsOpWriteValue(op, operand)) {
        affectedValues.push_back(operand);
      }
    }

    for (auto value : affectedValues) {
      for (auto user : value.getUsers()) {
        if (user == op) {
          continue;
        }
        dependencies.insert(user);
      }
    }
  }

  void identifyJointPaths(const SmallVector<SmallVector<Operation*>>& paths_a,
                          const SmallVector<SmallVector<Operation*>>& paths_b,
                          DenseSet<SmallVector<Operation*>>& joint_paths) {
    joint_paths.clear();
    for (const auto& path_a : paths_a) {
      for (const auto& path_b : paths_b) {
        SmallVector<Operation*> joint;
        for (int64_t i = 0; i < std::min(path_a.size(), path_b.size()); i++) {
          if (path_a[i] == path_b[i]) {
            joint.push_back[path_a[i]];
          } else {
            if (!joint.empty()) {
              joint_paths.insert(joint);
            }
            continue;
          }
        }
      }
    }
  }
}

bool DotGpuFusionStrategy::finalizeFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fused_pattern,
    SmallVectorImpl<Operation*>& excluded_ops) {
  auto& roots = fusion_pattern.getRootOps();

  // Currently, kDot fusion only support one root op.
  if (roots.size() == 1) {
    return true;
  }

  auto& op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());

  // Find path from ops in the fusion pattern to dominant.

  // {op, [paths from dom to op]}
  DenseMap<Operation*, SmallVector<SmallVector<Operation*>>> path_dom_to_op;
  path_dom_to_ops.emplace(dom, {dom});

  SmallVector<Operation*> worklist;
  Operation* dom = fusion_pattern.getDominantOp();
  worklist.push_back(dom);
  while (!worklist.empty()) {
    auto curr = worklist.back();
    worklist.pop_back();
    const auto& path_curr = path_dom_to_roots[curr];
    DenseSet<Operation*> dependencies;
    getDirectDependentOps(curr, dependencies);
    for (auto dep : dependencies) {
      if (!op_set.contains(dep)) {
        continue;
      }
      auto& path_dep = path_dom_to_op[dep];
      for (const auto& path : path_curr) {
        SmallVector<Operation*> new_path = path;
        new_path.push_back(dep);
        path_dep.emplace_back(std::move(new_path));
      }
      worklist.push_back(dep);
    }
  }

  // If paths are disjoint, remain the longest path. Otherwise, remain the
  // shortest joint part.

  DenseSet<SmallVector<Operation*>> joint_paths_of_roots;
  for (auto root_a : roots) {
    auto& paths_a = path_dom_to_ops[root_a];
    for (auto root_b : roots) {
      if (root_a == root_b) {
        continue;
      }
      auto& paths_b = path_dom_to_ops[root_b];

      DenseSet<SmallVector<Operation*>>& curr_joint_paths;
      identifyJointPaths(paths_a, paths_b, curr_joint_paths);
      joint_paths_of_roots.insert(curr_joint_paths.begin(),
                                  curr_joint_paths.end);
    }
  }

  // TODO: joint again. a function named remain joint.

  SmallVector<Operation*> final_path;
  if (joint_paths_of_roots.size() == 1 &&
      joint_paths_of_roots.begin()->size() == 1) {
    final_path = TODO;  // the longest path among all existing paths.
  } else {
    final_path = TODO:  // the shortest path among all joint paths.
  }

  // Finally, remove the ops whose path to dominant contains external ops.

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