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

#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

using placement_utils::kDiscShapeCalcAttr;

////////////////////// CPU SparseOp FusionStrategy Implemenation ////////////
////////////////////////////////////////////////////////////////////////

bool isOutputFusible(Operation* op) {
  return isa<lmhlo::DynamicReshapeOp>(op) || isa<lmhlo::DynamicGatherOp>(op);
}

bool SparseOpCpuFusionStrategy::isFusible(Operation* op) {
  return isa<lmhlo_disc::WhereOp>(op) || mlir::disc_ral::isFusible(op);
  // return isa<lmhlo_disc::WhereOp>(op) ||
  // isa<lmhlo::DynamicBroadcastInDimOp>(op) || isa<lmhlo::CompareOp>(op) ||
  // isOutputFusible(op);
}

bool SparseOpCpuFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern) {
  Operation* inferredDominantOp = nullptr;
  FusionType inferredFusionType = FusionType::kNone;
  SmallVector<Operation*> where_ops;
  SmallVector<Operation*> basic_fusible_ops;
  for (Operation* op : fusion_pattern.getOpList()) {
    if (this->isFusible(op)) {
      if (isa<lmhlo_disc::WhereOp>(op)) {
        where_ops.push_back(op);
      } else {
        basic_fusible_ops.push_back(op);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Fusion pattern should not contain any un-fusible ops\n");
      return false;
    }
  }

  // If it is a single supported fusible op which is not fused by base fusion
  // strategy, it may be fused with where later.
  if (where_ops.size() == 0 && basic_fusible_ops.size() >= 1) {
    return initFusionPatternBase(shapeAnalysis, fusion_pattern);
  }

  if (where_ops.size() != 1) {
    return false;
  }

  fusion_pattern.setFusionType(FusionType::kWhere);
  fusion_pattern.setDominantOp(where_ops[0]);

  return true;
}

bool SparseOpCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                        FusionPattern& lhs, FusionPattern& rhs,
                                        FusionPattern& target) {
  if (lhs.getFusionType() == FusionType::kWhere ||
      rhs.getFusionType() == FusionType::kWhere) {
    // make sure lhs and rhs are producer-consumer pair
    auto lhs_results = lhs.getResults();
    auto rhs_ops = rhs.getOpList();
    bool found = false;
    for (Value v : lhs_results) {
      for (Operation* user : getValueUsers(v)) {
        // user->dump();
        if (std::find(rhs_ops.begin(), rhs_ops.end(), user) != rhs_ops.end()) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      return false;
    }
  } else {
    return false;
  }

  // check fusiblity
  if (rhs.getFusionType() == FusionType::kWhere) {
    // Basic Input fusion for where op
    for (Operation* op : lhs.getOpList()) {
      if (!this->isFusible(op) || op->getAttr(kDiscShapeCalcAttr) != nullptr) {
        return false;
      }
    }
  } else if (lhs.getFusionType() == FusionType::kWhere) {
    // Output fusion with where op
    for (Operation* op : rhs.getOpList()) {
      if (!isOutputFusible(op)) {
        return false;
      }
    }
  } else {
    return false;
  }
  if (target.getOpList().size() >= 6) {
    llvm::dbgs() << "SparseOpCpuFusionStrategy::tryFuse success() \n";
    llvm::dbgs() << "*********************lhs*********************\n";
    dumpFusionPattern(lhs);
    llvm::dbgs() << "*********************rhs*********************\n";
    dumpFusionPattern(rhs);
    llvm::dbgs() << "*********************res*********************\n";
    dumpFusionPattern(target);
    llvm::dbgs() << "*********************end*********************\n\n";
  }
  return true;
}

}  // namespace disc_ral
}  // namespace mlir
