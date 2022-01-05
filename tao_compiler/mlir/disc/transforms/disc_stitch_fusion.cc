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
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace disc_ral {

namespace {

bool isOnGpu(Operation *op) {
  auto attr =
      op->getAttrOfType<StringAttr>(placement_utils::kDiscPlaceAssignment);
  if (attr) {
    return (attr.getValue() == placement_utils::kGpu);
  }

  // Fusion should have been set a placement attribute in lhlo_fusion pass.
  // Leave a default placement here just in case there are fusion ops not
  // generated by the fusion pass (e.g. in the file-check based ut).
  if (isa<lmhlo::FusionOp>(op))
    return true;

  assert(isa<lmhlo::LmhloOp>(op) && "Unexpected usage of isOnGpu");
  auto result_memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  auto memory_space =
      result_memref.getType().cast<MemRefType>().getMemorySpace();
  return memory_space && memory_space.isa<StringAttr>() &&
         memory_space.cast<StringAttr>().getValue() == placement_utils::kGpu;
}

LogicalResult HandleCpuFusionOp(OpBuilder &b, Operation *fusion) {
  auto fusionOp = cast<lmhlo::FusionOp>(fusion);
  assert(fusionOp);
  FusionPattern fusionPattern(fusionOp);
  if (!fusionPattern.isStitchFusion()) {
    // skip non-stitch fusion pattern.
    return success();
  }
  auto rootOps = fusionPattern.getRootOps();
  auto fusedBlock = &(fusionOp.region().front());

  // No need to do codegen, return directly.
  if (rootOps.empty()) {
    cleanUnusedLhloOps(fusedBlock);
    return success();
  }

  StitchCPUAnalysis stitchAnalysis(fusionPattern);
  if (!stitchAnalysis.doCodeGeneration(b, fusionOp)) {
    LLVM_DEBUG(llvm::dbgs() << "stitchAnalysis failed to doCodeGeneration\n");
    return failure();
  }

  return success();
}

struct DiscStitchFusion : public DiscStitchFusionBase<DiscStitchFusion> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    OpBuilder b(func);
    SmallVector<Operation *, 4> gpu_fusion_worklist;
    SmallVector<Operation *, 4> cpu_fusion_worklist;
    func.walk([&](lmhlo::FusionOp op) {
      if (isOnGpu(op))
        gpu_fusion_worklist.push_back(op);
      else
        cpu_fusion_worklist.push_back(op);
    });

    for (Operation *fusion : gpu_fusion_worklist) {
      // TODO(disc): handling stitch fusion on GPU.
      signalPassFailure();
      return;
    }

    for (Operation *fusion : cpu_fusion_worklist) {
      // Error message should be emitted inside the function.
      if (failed(HandleCpuFusionOp(b, fusion))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscStitchFusionPass() {
  return std::make_unique<DiscStitchFusion>();
}

} // namespace disc_ral
} // namespace mlir