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

#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

using placement_utils::kDiscShapeCalcAttr;

bool iskWhereOutputFusible(Operation* op) {
  return isa<lmhlo::DynamicReshapeOp>(op) || isa<lmhlo::DynamicGatherOp>(op);
}

bool iskWhereInputFusible(Operation* op) {
  return isa<lmhlo_disc::WhereOp>(op) ||
         isa<lmhlo::DynamicBroadcastInDimOp>(op) || isa<lmhlo::CompareOp>(op);
}

bool iskSparseReductionOutputFusible(Operation* op) {
  return isa<lmhlo::DynamicBroadcastInDimOp>(op) ||
         isa<lmhlo::DynamicReshapeOp>(op) || isa<lmhlo::SelectOp>(op);
}

bool isFusibleSparseReductionOp(Operation* op) {
  return isa<lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(op);
}

////////////////////// CPU SparseOp FusionStrategy Implemenation ////////////
////////////////////////////////////////////////////////////////////////

bool SparseOpCpuFusionStrategy::isFusible(Operation* op) {
  return isa<lmhlo_disc::WhereOp>(op) || isFusibleSparseReductionOp(op) ||
         mlir::disc_ral::isFusible(op);
}

bool SparseOpCpuFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern) {
  Operation* inferredDominantOp = nullptr;
  FusionType inferredFusionType = FusionType::kNone;
  SmallVector<Operation*> where_ops;
  SmallVector<Operation*> sparse_reduction_ops;
  SmallVector<Operation*> basic_fusible_ops;

  for (Operation* op : fusion_pattern.getOpList()) {
    if (this->isFusible(op)) {
      if (isa<lmhlo_disc::WhereOp>(op)) {
        where_ops.push_back(op);
      } else if (isFusibleSparseReductionOp(op)) {
        sparse_reduction_ops.push_back(op);
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
  // strategy, it may be fused later.
  if ((where_ops.size() == 0 && sparse_reduction_ops.size() == 0) &&
      basic_fusible_ops.size() >= 1) {
    return initFusionPatternBase(shapeAnalysis, fusion_pattern);
  }

  if (where_ops.size() > 1 || sparse_reduction_ops.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion pattern should not contain more than 1 "
                               "where op or sparse reduction op\n");
    return false;
  }

  if (where_ops.size() == 1 && sparse_reduction_ops.size() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion pattern should not contain where op and "
                               "sparse reduction op at the same time\n");
    return false;
  }

  if (where_ops.size() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "Init kWhere fusion \n");
    fusion_pattern.setFusionType(FusionType::kWhere);
    fusion_pattern.setDominantOp(where_ops[0]);
  } else if (sparse_reduction_ops.size() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "Init kSparseReduction fusion \n");
    fusion_pattern.setFusionType(FusionType::kSparseReduction);
    fusion_pattern.setDominantOp(sparse_reduction_ops[0]);
  }

  return true;
}

bool SparseOpCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                        FusionPattern& lhs, FusionPattern& rhs,
                                        FusionPattern& target) {
  if (mlir::disc_ral::isSparseFusion(lhs) ||
      mlir::disc_ral::isSparseFusion(rhs)) {
    // make sure lhs and rhs are producer-consumer pair
    auto lhs_results = lhs.getResults();
    auto rhs_ops = rhs.getOpList();
    bool found = false;
    for (Value v : lhs_results) {
      for (Operation* user : getValueUsers(v)) {
        if (std::find(rhs_ops.begin(), rhs_ops.end(), user) != rhs_ops.end()) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      LLVM_DEBUG(llvm::dbgs() << "rhs/lhs are not producer-consumer\n");
      return false;
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "rhs/lhs are both not sparse fusion, should be "
                               "handled by other FusionStrategy\n");
    return false;
  }

  // check fusiblity
  if (rhs.getFusionType() == FusionType::kWhere) {
    // Input fusion for where op
    for (Operation* op : lhs.getOpList()) {
      if (!iskWhereInputFusible(op) ||
          op->getAttr(kDiscShapeCalcAttr) != nullptr) {
        LLVM_DEBUG(llvm::dbgs()
                   << "tryFuse for kWhere input fusion failed, unfusible op or "
                      "fusible op with shape calculation attr encountered\n");
        return false;
      }
    }
  } else if (lhs.getFusionType() == FusionType::kSparseReduction) {
    // output fusion with sparse reduction op
    for (Operation* op : rhs.getOpList()) {
      if (!iskSparseReductionOutputFusible(op)) {
        return false;
      }
    }
  } else if (rhs.getFusionType() == FusionType::kSparseReduction) {
    // fuse kLoop that only broadcast constant
    for (Operation* op : lhs.getOpList()) {
      if (!(isa<lmhlo::DynamicBroadcastInDimOp>(op) ||
            isa<lmhlo::ConstantOp>(op))) {
        return false;
      }
    }
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "tryFuse for other fusion types is not supported now\n");
    return false;
  }
  return true;
}

}  // namespace disc_ral
}  // namespace mlir
