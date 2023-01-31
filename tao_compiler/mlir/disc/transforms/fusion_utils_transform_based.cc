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

namespace mlir {
namespace disc_ral {

/////////// Transform based CPU FusionStrategy Implemenation ///////////
////////////////////////////////////////////////////////////////////////

bool isSupportedDot(Operation* op) {
  auto dotOp = dyn_cast<lmhlo::DotGeneralOp>(op);
  if (!dotOp) return false;

  auto lhsTy = op->getOperand(0).getType().cast<MemRefType>();
  auto rhsTy = op->getOperand(1).getType().cast<MemRefType>();
  auto outTy = op->getOperand(2).getType().cast<MemRefType>();

  // TODO(wyzero): support batch gemm
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || outTy.getRank() != 2)
    return false;

  auto dimNumbers = dotOp.getDotDimensionNumbers();
  if (dimNumbers.getLhsBatchingDimensions().size() != 0 ||
      dimNumbers.getRhsBatchingDimensions().size() != 0)
    return false;

  auto lhsCntractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsCntractingDims = dimNumbers.getRhsContractingDimensions();
  return (lhsCntractingDims.size() == 1 && rhsCntractingDims.size() == 1);
}

bool TransformBasedCpuFusionStrategy::isFusible(Operation* op) {
  return isSupportedDot(op) || isa<lmhlo::ConstantOp>(op);
}

bool TransformBasedCpuFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusionPattern) {
  // firstly init the fusion kind to kNone
  fusionPattern.setDominantOp(nullptr);
  fusionPattern.setFusionType(FusionType::kNone);

  // special case for single operation.
  if (fusionPattern.getOpList().size() == 1) {
    Operation* op = *fusionPattern.getOpList().begin();
    if (this->isFusible(op)) {
      fusionPattern.setDominantOp(op);
      fusionPattern.setFusionType(FusionType::kTransform);
    }
    return true;
  }

  DenseSet<Value> dotWeights;
  DenseSet<Operation*> supportedDotOps;
  for (Operation* op : fusionPattern.getOpList()) {
    // early return for the case where there are non supported ops.
    if (!this->isFusible(op)) return true;
    if (isSupportedDot(op)) {
      supportedDotOps.insert(op);
      dotWeights.insert(op->getOperand(1));
    }
  }

  // Only support one gemm a.t.m.
  if (supportedDotOps.size() != 1) return true;

  // Only support fuse const ops that are used as weights for some dot ops and
  // not consumed by ops outside the fusion pattern.
  for (Operation* op : fusionPattern.getOpList()) {
    if (!isa<lmhlo::ConstantOp>(op)) continue;
    if (llvm::find(dotWeights, op->getOperand(0)) == dotWeights.end() ||
        llvm::find(fusionPattern.getRootOps(), op) !=
            fusionPattern.getRootOps().end())
      return true;
  }

  fusionPattern.setDominantOp(*supportedDotOps.begin());
  fusionPattern.setFusionType(FusionType::kTransform);
  return true;
}

bool TransformBasedCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                              FusionPattern& lhs,
                                              FusionPattern& rhs,
                                              FusionPattern& target) {
  if (!initFusionPattern(shapeAnalysis, target)) return false;
  return target.isTransformBasedFusion();
}

}  // namespace disc_ral
}  // namespace mlir