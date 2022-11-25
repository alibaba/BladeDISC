/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/transforms/passes.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

// This file implements logic to legalize transform fusion pattern to loop.

namespace mlir {
namespace disc_ral {

namespace {

BlockAndValueMapping buildValueMapping(FusionPattern& fusionPattern,
                                       func::FuncOp funcOp,
                                       bool reverse = false) {
  int argPosition = 0;
  BlockAndValueMapping mapping;
  auto addMapping = [&](Value v) {
    if (reverse)
      mapping.map(funcOp.getArgument(argPosition++), v);
    else
      mapping.map(v, funcOp.getArgument(argPosition++));
  };

  for (Value v : fusionPattern.getOperands()) addMapping(v);
  for (Value v : fusionPattern.getInternalResults()) addMapping(v);
  for (Value v : fusionPattern.getResults()) addMapping(v);
  return mapping;
}

struct DiscTransformLegalizeToLoopPass
    : public DiscTransformLegalizeToLoopPassBase<
          DiscTransformLegalizeToLoopPass> {
  explicit DiscTransformLegalizeToLoopPass(bool gpuEnabled,
                                           const std::string& transformFileName,
                                           bool enableExpensiveChecks)
      : DiscTransformLegalizeToLoopPassBase<DiscTransformLegalizeToLoopPass>::
            DiscTransformLegalizeToLoopPassBase() {
    this->gpuEnabled_ = gpuEnabled;
    this->transformFileName_ = transformFileName;
    this->enableExpensiveChecks_ = enableExpensiveChecks;
  }

  void runOnOperation() override;

  LogicalResult handleCpuFusionOp(OpBuilder& b, Operation* fusion,
                                  ShapeAnalysis& shapeAnalysis);
  // Outlines the fusion op to a standalone module op.
  LogicalResult outlineFusionOp(lmhlo::FusionOp fusionOp,
                                FusionPattern& fusionPattern,
                                OwningOpRef<ModuleOp>& m);
  // Builds a nested pass pipeline to legalize the outlined fusion op.
  LogicalResult runTransformPipeline(ModuleOp m);
  // Inlines the lowered IR into the orignal module.
  LogicalResult inlineTransformedModule(OpBuilder& b, Operation* fusion,
                                        FusionPattern& fusionPattern,
                                        ModuleOp m);
};

LogicalResult DiscTransformLegalizeToLoopPass::outlineFusionOp(
    lmhlo::FusionOp fusionOp, FusionPattern& fusionPattern,
    OwningOpRef<ModuleOp>& m) {
  Location loc = fusionOp->getLoc();
  m = ModuleOp::create(loc);
  auto b = OpBuilder::atBlockBegin(m->getBody());

  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  for (Value v : fusionPattern.getOperands()) {
    inputTypes.push_back(v.getType());
  }
  for (Value v : fusionPattern.getInternalResults()) {
    inputTypes.push_back(v.getType());
  }
  for (Value v : fusionPattern.getResults()) {
    inputTypes.push_back(v.getType());
    outputTypes.push_back(v.getType());
  }
  auto funcType =
      FunctionType::get(fusionOp->getContext(), inputTypes, outputTypes);
  auto funcOp =
      b.create<func::FuncOp>(loc, getFusionFullName(fusionOp), funcType);
  Block* entryBlock = funcOp.addEntryBlock();
  auto mapping = buildValueMapping(fusionPattern, funcOp);
  b.setInsertionPoint(entryBlock, entryBlock->begin());
  b.clone(*fusionOp.getOperation(), mapping);
  b.create<func::ReturnOp>(loc, funcOp.getArguments().drop_front(
                                    fusionPattern.getOperands().size() +
                                    fusionPattern.getInternalResults().size()));
  return success();
}

LogicalResult DiscTransformLegalizeToLoopPass::runTransformPipeline(
    ModuleOp m) {
  PassManager pm(m.getContext());
  pm.addPass(createDiscLegalizeLmhloFusionToLinalgPass());
  pm.addPass(createDiscTransformDialectInterpreterPass(transformFileName_,
                                                       enableExpensiveChecks_));
  pm.addPass(createDiscRewritePayloadIRForRALPass(gpuEnabled_));
  return pm.run(m);
}

LogicalResult DiscTransformLegalizeToLoopPass::inlineTransformedModule(
    OpBuilder& b, Operation* fusion, FusionPattern& fusionPattern, ModuleOp m) {
  auto funcOps = llvm::to_vector<4>(m.getOps<func::FuncOp>());
  if (funcOps.size() != 1)
    return fusion->emitError() << "failed to inline the transformed module "
                                  "with multiple functions\n";
  if (funcOps[0].getBody().getBlocks().size() != 1)
    return fusion->emitError()
           << "failed to inline the transformed func with multiple blocks\n";

  Block* body = cast<lmhlo::FusionOp>(fusion).getBody();
  for (auto& nestedOp : llvm::make_early_inc_range(body->without_terminator()))
    nestedOp.erase();
  b.setInsertionPoint(body, body->begin());
  auto mapping = buildValueMapping(fusionPattern, funcOps[0], true);
  for (auto& nestedOp : funcOps[0].getBody().front().without_terminator()) {
    Operation* cloned = b.clone(nestedOp, mapping);
  }
  return success();
}

LogicalResult DiscTransformLegalizeToLoopPass::handleCpuFusionOp(
    OpBuilder& b, Operation* fusion, ShapeAnalysis& shapeAnalysis) {
  auto fusionOp = cast<lmhlo::FusionOp>(fusion);
  assert(fusionOp);
  FusionPattern fusionPattern(fusionOp, &shapeAnalysis);
  if (!fusionPattern.isTransformBasedFusion()) {
    // skip non-transform-based fusion pattern.
    return success();
  }

  // 1, Outline the fusion to a standalone module op.
  OwningOpRef<ModuleOp> m;
  if (failed(outlineFusionOp(fusionOp, fusionPattern, m))) return failure();
  LLVM_DEBUG(llvm::dbgs() << "After outline fusion op:\n" << m.get() << "\n");

  // 2, TODO(wyzero): assign a default schedule for each pattern here.

  // 3, Build a nested pass pipeline to legalize the outlined fusion op.
  if (failed(runTransformPipeline(m.get()))) return failure();
  LLVM_DEBUG(llvm::dbgs() << "After run transform pipeline:\n"
                          << m.get() << "\n");

  // 4, Inline the lowered IR into the orignal module.
  if (failed(inlineTransformedModule(b, fusion, fusionPattern, m.get())))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "After inline transformed module:\n"
                          << *fusion << "\n");
  return success();
}

void DiscTransformLegalizeToLoopPass::runOnOperation() {
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
    return signalPassFailure();
  }

  for (Operation* fusion : cpu_fusion_worklist) {
    if (!useShapeConstraintIR()) {
      // TODO: use FuncOp that contains `fusionOp` to construct
      // shape-analysis, which will use global information for shape equality
      // and decomposition analysis.
      shapeAnalysisPtr.reset(new ShapeAnalysisDeprecated{fusion});
    }

    // Error message should be emitted inside the function.
    if (failed(handleCpuFusionOp(b, fusion, *shapeAnalysisPtr))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscTransformLegalizeToLoopPass(bool gpuEnabled,
                                      const std::string& filename,
                                      bool expensiveCheck) {
  return std::make_unique<DiscTransformLegalizeToLoopPass>(gpuEnabled, filename,
                                                           expensiveCheck);
}

}  // namespace disc_ral
}  // namespace mlir
