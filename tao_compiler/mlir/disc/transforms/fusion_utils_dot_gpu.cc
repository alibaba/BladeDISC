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

#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/utils/source_emitter.h"

namespace llvm {
template <>
struct DenseMapInfo<SmallVector<mlir::Operation*>> {
  static SmallVector<mlir::Operation*> getEmptyKey() {
    return SmallVector<mlir::Operation*>{
        DenseMapInfo<mlir::Operation*>::getEmptyKey()};
  }

  static SmallVector<mlir::Operation*> getTombstoneKey() {
    return SmallVector<mlir::Operation*>{
        DenseMapInfo<mlir::Operation*>::getTombstoneKey()};
  }

  static unsigned getHashValue(const SmallVector<mlir::Operation*>& vs) {
    unsigned hash = hash_value(vs.size());
    for (auto v : vs) {
      hash = llvm::hash_combine(hash, v);
    }
    return hash;
  }

  static bool isEqual(const SmallVector<mlir::Operation*>& lhs,
                      const SmallVector<mlir::Operation*>& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

namespace mlir {
namespace disc_ral {

namespace {

void getDirectConsumerOps(Operation* op, DenseSet<Operation*>& consumers) {
  consumers.clear();
  if (op == nullptr) {
    return;
  }
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
      consumers.insert(user);
    }
  }
}

void getDirectProducerOpsInFusionPattern(Operation* op,
                                         FusionPattern& fusion_pattern,
                                         DenseSet<Operation*>& producers) {
  producers.clear();

  auto& op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());

  int64_t num_input_operand = op->getNumOperands() - getNumResultOperands(op);
  for (Value value : op->getOperands().take_front(num_input_operand)) {
    auto producer = fusion_pattern.findLastWriter(value);
    if (op_set.contains(producer)) {
      producers.insert(producer);
    }
  }
}

}  // namespace

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

void getValueWritter(Value value, SmallVectorImpl<Operation*>& writter) {
  for (auto user : value.getUsers()) {
    if (IsOpWriteValue(user, value)) {
      writter.push_back(user);
    }
  }
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
  /*
   * 1. Only one dot is supported per fusion, currently.
   * 2. Dot is the first op;
   * 3. Other ops' effective operands are not the operands of the fusion,
   *    currently; otherwise this op should be scalar/splat constant.
   *
   * The conditions 1 and 3 will be loosed in the future.
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

  // All ops are supported by CUDA source emitter. The checking is necessary
  // because this function may called without the calling of isFusible.
  for (auto op : mem_intensive_ops) {
    if (!isFusible(op)) {
      return false;
    }
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

  // There should be at most 1 transpose op, meeting the following requirement:
  // 1. all paths from roots to the dot op contains the transpose op;
  // 2. the input dims of the transpose op are the same with that of the dot op.
  // 3. currently, only support permutation of [0,2,1,3], for which dime-1
  //    should be static-shaped.

  // At most one transpose op.
  Operation* transpose = nullptr;
  for (auto op : op_list) {
    if (isa<lmhlo::TransposeOp>(op)) {
      if (transpose != nullptr) {
        return false;
      } else {
        transpose = op;
      }
    }
  }

  if (transpose != nullptr) {
    // Check all paths from dom to roots for transpose op.
    SmallVector<SmallVector<Operation*>> all_paths_dom_to_roots;
    DenseMap<Operation*, DenseSet<SmallVector<Operation*>>> path_dom_to_ops;
    Operation* dom = fusion_pattern.getDominantOp();
    DenseSet<SmallVector<Operation*>> path_to_dom{SmallVector<Operation*>{dom}};
    path_dom_to_ops.try_emplace(dom, std::move(path_to_dom));
    SmallVector<Operation*> worklist;
    worklist.push_back(dom);
    auto roots = fusion_pattern.getRootOps();
    DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
    while (!worklist.empty()) {
      auto curr = worklist.back();
      worklist.pop_back();
      const auto& path_curr = path_dom_to_ops[curr];
      DenseSet<Operation*> consumers;
      getDirectConsumerOps(curr, consumers);
      for (auto consumer : consumers) {
        if (!op_set.contains(consumer)) {
          continue;
        }
        auto& path_consumer = path_dom_to_ops[consumer];
        for (const auto& path : path_curr) {
          SmallVector<Operation*> new_path = path;
          new_path.push_back(consumer);
          path_consumer.insert(std::move(new_path));
        }
        worklist.push_back(consumer);
      }
    }
    for (auto root : roots) {
      auto& path = path_dom_to_ops[root];
      all_paths_dom_to_roots.insert(all_paths_dom_to_roots.end(), path.begin(),
                                    path.end());
    }
    for (auto& path : all_paths_dom_to_roots) {
      if (llvm::find(path, transpose) == path.end()) {
        return false;
      }
    }

    // Make sure that the input dims of the transpose op are the same with the
    // output of the dot op.
    if (!shapeAnalysis.isShapeEqual(dot_ops[0]->getOperand(2),
                                    transpose->getOperand(0))) {
      return false;
    }

    // Only support permutation of [0,2,1,3], and dim-1 is static-shaped.
    auto perm_attr = dyn_cast<lmhlo::TransposeOp>(transpose)
                         .getPermutation()
                         .getValues<int64_t>();
    SmallVector<int64_t> perm{perm_attr.begin(), perm_attr.end()};
    if (!(perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3)) {
      return false;
    }
    auto memref_ty = dot_ops[0]->getOperand(2).getType().dyn_cast<ShapedType>();
    if (!memref_ty) {
      return false;
    }
    auto dim_1 = memref_ty.getDimSize(1);
    if (dim_1 == ShapedType::kDynamic) {
      return false;
    }
  }

  fusion_pattern.setFusionType(FusionType::kDot);
  fusion_pattern.setDominantOp(dot_ops[0]);

  return true;
}

bool DotGpuFusionStrategy::pruneFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern,
    SmallVectorImpl<Operation*>& excluded_ops) {
  if (fusion_pattern.getFusionType() != FusionType::kDot) {
    // None of my business.
    return true;
  }

  // Currently, kDot fusion only support one root op. If there are many roots,
  // find the joint paths between roots, and remain the shortest. This is to
  // guarantee that there is only one output of the fusion. We will support
  // multi-output fusion in the future.

  auto roots = fusion_pattern.getRootOps();
  if (roots.size() == 1) {
    return true;
  }

  auto op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());

  SmallVector<Operation*> dots;
  for (auto op : op_list) {
    if (isa<lmhlo::DotGeneralOp>(op)) {
      dots.push_back(op);
    }
  }
  if (dots.empty()) {
    // Not kDot fusion. Do not deal with it.
    return true;
  }

  // Find path from dominant to other ops in the fusion pattern.

  // {op, {paths from dom to op}}
  DenseMap<Operation*, DenseSet<SmallVector<Operation*>>> path_dom_to_ops;
  Operation* dom = fusion_pattern.getDominantOp();
  DenseSet<SmallVector<Operation*>> path_to_dom{SmallVector<Operation*>{dom}};
  path_dom_to_ops.try_emplace(dom, std::move(path_to_dom));

  SmallVector<Operation*> worklist;
  worklist.push_back(dom);
  while (!worklist.empty()) {
    auto curr = worklist.back();
    worklist.pop_back();
    const auto& path_curr = path_dom_to_ops[curr];
    DenseSet<Operation*> consumers;
    getDirectConsumerOps(curr, consumers);
    for (auto consumer : consumers) {
      if (!op_set.contains(consumer)) {
        continue;
      }
      auto& path_consumer = path_dom_to_ops[consumer];
      for (const auto& path : path_curr) {
        SmallVector<Operation*> new_path = path;
        new_path.push_back(consumer);
        path_consumer.insert(std::move(new_path));
      }
      worklist.push_back(consumer);
    }
  }

  // Find the shortest joint path to all roots.
  SmallVector<Operation*> shortest_path;
  SmallVector<SmallVector<Operation*>> all_paths_to_all_roots;
  for (auto root : roots) {
    auto& path = path_dom_to_ops[root];
    all_paths_to_all_roots.insert(all_paths_to_all_roots.end(), path.begin(),
                                  path.end());
  }
  std::size_t min_size = op_list.size();
  for (auto& path : all_paths_to_all_roots) {
    min_size = std::min(min_size, path.size());
  }
  for (std::size_t joint_path_length = 0; joint_path_length < min_size;
       joint_path_length++) {
    bool same = true;
    auto op_at_this_length = all_paths_to_all_roots[0][joint_path_length];
    for (int64_t n = 1; n < all_paths_to_all_roots.size(); n++) {
      if (op_at_this_length != all_paths_to_all_roots[n][joint_path_length]) {
        same = false;
        break;
      }
    }
    if (!same) {
      break;
    } else {
      shortest_path.push_back(op_at_this_length);
    }
  }

  // Form the new fusion pattern.
  auto new_root = shortest_path.back();
  worklist.clear();
  worklist.push_back(new_root);
  DenseSet<Operation*> effective_ops_set;
  effective_ops_set.insert(new_root);
  while (!worklist.empty()) {
    auto curr = worklist.back();
    worklist.pop_back();
    DenseSet<Operation*> direct_producers;
    getDirectProducerOpsInFusionPattern(curr, fusion_pattern, direct_producers);
    worklist.insert(worklist.end(), direct_producers.begin(),
                    direct_producers.end());
    effective_ops_set.insert(direct_producers.begin(), direct_producers.end());
  }
  SmallVector<Operation*> effective_ops(effective_ops_set.begin(),
                                        effective_ops_set.end());
  FusionPattern new_fusion_pattern =
      FusionPattern::createWithoutInit(effective_ops);
  initFusionPattern(shapeAnalysis, new_fusion_pattern);
  fusion_pattern = std::move(new_fusion_pattern);

  // Finally, identify the ops not consumed by the new root op.
  excluded_ops.clear();
  for (auto op : op_list) {
    if (!effective_ops_set.contains(op)) {
      excluded_ops.push_back(op);
    }
  }

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
  if (isa<lmhlo::DynamicBroadcastInDimOp>(op)) {
    if (!SourceEmitterCUDA::isBroadcastOnScalarOrSplatConstant(op)) {
      effective_operands.push_back(op->getOperand(0));
    }
  } else if (isa<lmhlo::DynamicReshapeOp, lmhlo::TransposeOp>(op)) {
    effective_operands.push_back(op->getOperand(0));
  } else {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    effective_operands = op->getOperands().take_front(num_input_operand);
  }
  return effective_operands;
}

}  // namespace disc_ral
}  // namespace mlir