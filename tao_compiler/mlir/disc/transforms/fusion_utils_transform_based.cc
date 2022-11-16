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

namespace mlir {
namespace disc_ral {

/////////// Transform based CPU FusionStrategy Implemenation ///////////
////////////////////////////////////////////////////////////////////////

bool TransformBasedCpuFusionStrategy::isFusible(Operation* op) {
  // Only support matmul a.t.m.
  // TODO(wyzero): support const weight
  // TODO(wyzero): support gemm + elemwise fusion
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

  // TODO(wyzero): support dot_general with fused transpose
  auto lhsCntractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsCntractingDims = dimNumbers.getRhsContractingDimensions();
  return (lhsCntractingDims.size() == 1 && lhsCntractingDims[0] == 1 &&
          rhsCntractingDims.size() == 1 && rhsCntractingDims[0] == 0);
}

bool TransformBasedCpuFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusionPattern) {
  Operation* inferredDominantOp = nullptr;
  FusionType inferredFusionType = FusionType::kNone;

  // TODO(wyzero): support dot fusion
  bool allSupported =
      llvm::all_of(fusionPattern.getOpList(),
                   [&](Operation* op) { return this->isFusible(op); });
  if (allSupported && fusionPattern.getOpList().size() == 1) {
    inferredDominantOp = fusionPattern.getOpList()[0];
    inferredFusionType = FusionType::kTransform;
  }
  fusionPattern.setDominantOp(inferredDominantOp);
  fusionPattern.setFusionType(inferredFusionType);
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