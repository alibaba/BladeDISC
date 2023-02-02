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

// This file implements logic for lowering LHLO dialect to Affine dialect.
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

LogicalResult HandleCpuFusionOp(OpBuilder& b, Operation* fusion,
                                ShapeAnalysis& shapeAnalysis) {
  auto fusionOp = cast<lmhlo::FusionOp>(fusion);
  assert(fusionOp);
  FusionPattern fusionPattern(fusionOp, &shapeAnalysis);
  if (!fusionPattern.isStitchFusion()) {
    // skip non-stitch fusion pattern.
    return success();
  }
  auto rootOps = fusionPattern.getRootOps();
  auto fusedBlock = &(fusionOp.getRegion().front());

  // No need to do codegen, return directly.
  if (rootOps.empty()) {
    cleanUnusedLhloOps(fusedBlock);
    return success();
  }

  StitchCPUAnalysis stitchAnalysis(fusionPattern, shapeAnalysis);
  if (!stitchAnalysis.doCodeGeneration(b, fusionOp)) {
    LLVM_DEBUG(llvm::dbgs() << "stitchAnalysis failed to doCodeGeneration\n");
    return failure();
  }

  return success();
}

struct DiscStitchFusion : public DiscStitchFusionBase<DiscStitchFusion> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder b(func);
    SmallVector<Operation*, 4> gpu_fusion_worklist;
    SmallVector<Operation*, 4> cpu_fusion_worklist;
    func.walk([&](lmhlo::FusionOp op) {
      if (placement_utils::isGpuLmhlo(op))
        gpu_fusion_worklist.push_back(op);
      else
        cpu_fusion_worklist.push_back(op);
    });

    std::unique_ptr<ShapeAnalysis> shapeAnalysisPtr;
    if (!gpu_fusion_worklist.empty() || !cpu_fusion_worklist.empty()) {
      if (useShapeConstraintIR()) {
        shapeAnalysisPtr.reset(new ShapeConstraintIRAnalysis(func));
      }
    }

    for (Operation* fusion : gpu_fusion_worklist) {
      // TODO(disc): handling stitch fusion on GPU.
      signalPassFailure();
      return;
    }

    for (Operation* fusion : cpu_fusion_worklist) {
      if (!useShapeConstraintIR()) {
        // TODO: use FuncOp that contains `fusionOp` to construct
        // shape-analysis, which will use global information for shape equality
        // and decomposition analysis.
        shapeAnalysisPtr.reset(new ShapeAnalysisDeprecated{fusion});
      }

      // Error message should be emitted inside the function.
      if (failed(HandleCpuFusionOp(b, fusion, *shapeAnalysisPtr))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscStitchFusionPass() {
  return std::make_unique<DiscStitchFusion>();
}

}  // namespace disc_ral
}  // namespace mlir
