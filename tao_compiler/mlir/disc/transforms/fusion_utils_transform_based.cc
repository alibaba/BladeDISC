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

bool isScalarConstOp(Operation* op) {
  auto constOp = dyn_cast<lmhlo::ConstantOp>(op);
  if (!constOp) return false;
  MemRefType type = constOp.getOutput().getType().cast<MemRefType>();
  return (type.getRank() == 0 || constOp.getValue().isSplat());
}

bool isBcastOp(Operation* op) {
  return isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
             lmhlo::DynamicBroadcastInDimOp>(op);
}

bool isSupportedBcast(Operation* op, ShapeAnalysis& shapeAnalysisBase) {
  if (!isBcastOp(op)) return false;
  auto shapeIRAnalysis =
      dynamic_cast<ShapeConstraintIRAnalysis*>(&shapeAnalysisBase);
  auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
  assert(dimAttr);
  auto dimensions = dimAttr.getValues<int64_t>();
  auto in = op->getOperand(0);
  auto inType = in.getType().cast<MemRefType>();
  auto out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  if (inType.getRank() != dimensions.size()) return false;
  for (auto [inDimIdx, inDimSize] : llvm::enumerate(inType.getShape())) {
    int64_t outDimIdx = dimensions[inDimIdx];
    if (inDimSize != ShapedType::kDynamic) continue;
    // linalg generic op does not support "runtime broadcast semantic", thus we
    // have to know if we need to broadcast in the compile time.
    if (!shapeIRAnalysis ||
        !shapeIRAnalysis->isProductEqual(in, {inDimIdx}, out, {outDimIdx}))
      return false;
  }
  return true;
}

bool TransformBasedCpuFusionStrategy::isFusible(Operation* op) {
  if (!useTransformGEMMEpilogueFusionSchedule()) {
    return isSupportedDot(op) || isa<lmhlo::ConstantOp>(op);
  }
  return isSupportedDot(op) || isElementWise(op) || isBcastOp(op) ||
         isa<lmhlo::ConstantOp>(op);
}

bool TransformBasedCpuFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusionPattern) {
  // firstly init the fusion kind to kNone
  fusionPattern.setDominantOp(nullptr);
  fusionPattern.setFusionType(FusionType::kNone);

  // special case for single operation.
  if (fusionPattern.getOpList().size() == 1) {
    Operation* op = *fusionPattern.getOpList().begin();
    if (!isBcastOp(op) && this->isFusible(op) ||
        isBcastOp(op) && isSupportedBcast(op, shapeAnalysis)) {
      fusionPattern.setDominantOp(op);
      fusionPattern.setFusionType(isSupportedDot(op) ? FusionType::kTransform
                                                     : FusionType::kLoop);
    }
    return true;
  }

  // We only support single output right now
  if (fusionPattern.getResults().size() != 1) return true;

  DenseSet<Value> dotWeights;
  DenseSet<Operation*> supportedDotOps;
  for (Operation* op : fusionPattern.getOpList()) {
    // early return for the case where there are non supported ops.
    if (!this->isFusible(op) ||
        isBcastOp(op) && !isSupportedBcast(op, shapeAnalysis))
      return true;
    if (isSupportedDot(op)) {
      supportedDotOps.insert(op);
      dotWeights.insert(op->getOperand(1));
    }
  }

  // Only support at most one gemm a.t.m.
  if (supportedDotOps.size() > 1) return true;
  if (supportedDotOps.empty()) {
    // special case: for elem+bcast epilogue subgraph fusion
    // 1, check no large const
    if (llvm::all_of(fusionPattern.getOpList(), [&](Operation* op) {
          return !isa<lmhlo::ConstantOp>(op) || isScalarConstOp(op);
        })) {
      fusionPattern.setFusionType(FusionType::kLoop);
      fusionPattern.setDominantOp(*fusionPattern.getRootOps().begin());
    }
    return true;
  }

  // normal case: gemm + epilogue fusion
  Operation* dominantDotOp = *supportedDotOps.begin();

  // Only support fuse const ops that are not consumed by ops outside the fusion
  // pattern and have one of the following properties:
  // - const ops that are used as weights for some dot ops
  // - const op has single element.
  DenseSet<Value> constDotWeights;
  for (Operation* op : fusionPattern.getOpList()) {
    if (!isa<lmhlo::ConstantOp>(op)) continue;
    if (llvm::find(fusionPattern.getRootOps(), op) !=
        fusionPattern.getRootOps().end())
      return true;
    if (llvm::find(dotWeights, op->getOperand(0)) != dotWeights.end()) {
      constDotWeights.insert(op->getOperand(0));
      continue;
    }
    if (!isScalarConstOp(op)) return true;
  }

  // We only support epilogue fusion right now.
  // Check if the dot op does not consume any result produced by other
  // lmhlo ops (except const ops).
  for (Value operand : dominantDotOp->getOperands().drop_back()) {
    if (llvm::find(constDotWeights, operand) != constDotWeights.end()) continue;
    auto& operands = fusionPattern.getOperands();
    if (llvm::find(operands, operand) == operands.end()) return true;
  }

  // the shape of the output should be the same as the shape of result of
  // dominant op.
  for (Value result : fusionPattern.getResults()) {
    if (!shapeAnalysis.isShapeEqual(result, dominantDotOp->getOperand(2)))
      return true;
  }

  fusionPattern.setDominantOp(dominantDotOp);
  fusionPattern.setFusionType(FusionType::kTransform);
  return true;
}

bool TransformBasedCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                              FusionPattern& lhs,
                                              FusionPattern& rhs,
                                              FusionPattern& target) {
  if (!initFusionPattern(shapeAnalysis, target)) return false;
  return target.isFusible();
}

}  // namespace disc_ral
}  // namespace mlir