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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/transforms/passes.h"
#include "mlir/disc/tools/disc-transform/utils.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/disc_transform_schedule.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/tsl/platform/default/logging.h"

// This file implements logic to legalize transform fusion pattern to loop.

namespace mlir {
namespace disc_ral {

namespace {

// map<pattern-name, filename>, bypass codegen pass pipeline and use IR passed
// from `filename`.
const std::unordered_map<std::string, std::string>&
bypassCodegenPatternNameMap() {
  static std::unordered_map<std::string, std::string> m = []() {
    std::unordered_map<std::string, std::string> m;
    const char* env = getenv("DISC_TRANSFORM_DEBUG_BYPASS_FUSION_PATTERNS");
    if (env) {
      SmallVector<StringRef> settings;
      StringRef(env).split(settings, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (auto& setting : settings) {
        SmallVector<StringRef> items;
        setting.split(items, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        assert(setting.size() == 2);
        m[items[0].str()] = items[1].str();
      }
    }
    return m;
  }();
  return m;
}

IRMapping buildValueMapping(FusionPattern& fusionPattern, func::FuncOp funcOp,
                            bool reverse = false) {
  int argPosition = 0;
  IRMapping mapping;
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

  void getDependentDialects(DialectRegistry& registry) const override {
    addTransformDialectDependentDialects(registry);
  }

  LogicalResult handleCpuFusionOp(OpBuilder& b, Operation* fusion,
                                  ShapeAnalysis& shapeAnalysis,
                                  ScheduleDispatcher& scheduleDispatcher);

  // Inject schedule selection logic
  LogicalResult injectScheduleSelectionIR(
      OpBuilder& b, PatternDescription& pd,
      SmallVectorImpl<Operation*>& clonedOps);

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
  if (VLOG_IS_ON(1))
    llvm::dbgs() << "/// ------- Apply Transform IR for fusion:\n"
                 << m << "\n\n";

  PassManager pm(m.getContext());
  auto printingFlags = OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](Pass* pass, Operation*) { return VLOG_IS_ON(1); },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(createDiscLegalizeLmhloFusionToLinalgPass());
  // Using transform IR attached in the module.
  pm.addPass(createDiscTransformDialectInterpreterPass(
      /* transformFileName */ "", enableExpensiveChecks_));
  pm.addPass(createDiscTransformDialectEraseSchedulePass());
  pm.addNestedPass<func::FuncOp>(createDiscMemrefCopyToLinalgPass());
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
    // ConstantWrapperOp will be cloned when we handle its corresponding
    // bufferization::to_memref op.
    if (isa<disc_linalg_ext::ConstantWrapperOp>(&nestedOp)) continue;
    if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(&nestedOp)) {
      auto constOp = toMemrefOp->getOperand(0)
                         .getDefiningOp<disc_linalg_ext::ConstantWrapperOp>();
      if (!constOp)
        return constOp->emitError()
               << "unkown operand for bufferization::ToMemrefOp\n";
      auto ip = b.saveInsertionPoint();
      b.setInsertionPoint(fusion);
      Location loc = constOp->getLoc();
      Value buffer = b.create<memref::AllocOp>(
          loc, toMemrefOp.getResult().getType().cast<MemRefType>());
      b.create<lmhlo::ConstantOp>(loc, constOp.getValue(), buffer);
      mapping.map(toMemrefOp.getResult(), buffer);
      b.restoreInsertionPoint(ip);
      continue;
    }
    Operation* cloned = b.clone(nestedOp, mapping);
  }
  return success();
}

LogicalResult DiscTransformLegalizeToLoopPass::injectScheduleSelectionIR(
    OpBuilder& b, PatternDescription& pd,
    SmallVectorImpl<Operation*>& clonedOps) {
  auto fusionOp = pd.getFusionOp();
  auto factories =
      ScheduleFactoryRegistry::get().getAllCandidateScheduleFactories(pd);
  if (factories.empty()) {
    return fusionOp->emitError() << "failed to find candidate schedule.\n";
  }
  // The returned candidate factories are sorted by priority.
  // The last schedule should always have `noGuardCondition`.
  if (!factories.back()->noGuardCondition(pd)) {
    return fusionOp->emitError()
           << "There are no canidate schedules for some shapes of "
           << pd.getTaggedPatternStr() << "\n";
  }
  // Only one candidate, no need to inject selection logic.
  if (factories.size() == 1) {
    mergeFusionTag(b, fusionOp, factories.back()->getTagSet());
    clonedOps.push_back(fusionOp);
    return success();
  }
  for (size_t i = 0; i < factories.size() - 1; ++i) {
    auto cloned = cast<lmhlo::FusionOp>(b.clone(*fusionOp.getOperation()));
    mergeFusionTag(b, cloned, factories[i]->getTagSet());
    FusionPattern fusionPattern(cloned, &pd.getShapeAnalysis());
    PatternDescription clonedPd(cloned, fusionPattern, pd.getShapeAnalysis());
    Value pred;
    if (failed(factories[i]->buildGuardCondition(b, cloned->getLoc(), clonedPd,
                                                 pred)))
      return cloned->emitError() << "faield to build guard IR\n";

    auto ifOp = b.create<scf::IfOp>(cloned->getLoc(), llvm::None, pred, true);
    cloned->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->begin());
    fusionOp->moveBefore(ifOp.elseBlock(), ifOp.elseBlock()->begin());
    clonedOps.push_back(cloned);
    b.setInsertionPointToStart(ifOp.elseBlock());
  }
  mergeFusionTag(b, fusionOp, factories.back()->getTagSet());
  clonedOps.push_back(fusionOp);
  return success();
}

LogicalResult DiscTransformLegalizeToLoopPass::handleCpuFusionOp(
    OpBuilder& b, Operation* fusion, ShapeAnalysis& shapeAnalysis,
    ScheduleDispatcher& scheduleDispatcher) {
  b.setInsertionPoint(fusion);
  auto fusionOp = cast<lmhlo::FusionOp>(fusion);
  assert(fusionOp);
  FusionPattern fusionPattern(fusionOp, &shapeAnalysis);
  if (!fusionPattern.isTransformBasedFusion()) {
    // skip non-transform-based fusion pattern.
    return success();
  }
  auto& bypassMap = bypassCodegenPatternNameMap();
  auto it = bypassMap.find(getFusionName(fusionOp).str());
  if (it != bypassMap.end()) {
    OwningOpRef<ModuleOp> m;
    if (failed(parseTransformModuleFromFile(b.getContext(), it->second, m))) {
      llvm::dbgs() << "illegal bypass transform fusion pattern codegen "
                      "setting, unable to load module from: "
                   << it->second << "\n";
      return failure();
    }
    // Inline the lowered IR into the orignal module.
    if (failed(inlineTransformedModule(b, fusion, fusionPattern, m.get()))) {
      return fusion->emitError()
             << "failed to inline module load from bypass setting\n";
    }
    return success();
  }

  // 0, inject schedule selection logic
  // clone the fusion op, each for one candidate schedule.
  SmallVector<Operation*> clonedFusionOps;
  PatternDescription pd(fusionOp, fusionPattern, shapeAnalysis);
  if (failed(injectScheduleSelectionIR(b, pd, clonedFusionOps))) {
    return fusionOp->emitError() << "failed to injectScheduleSelectionIR\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "After injectScheduleSelectionIR:\n"
                          << fusion->getParentOfType<func::FuncOp>() << "\n");

  for (auto fusion : clonedFusionOps) {
    b.setInsertionPoint(fusion);
    auto fusionOp = cast<lmhlo::FusionOp>(fusion);
    FusionPattern fusionPattern(fusionOp, &shapeAnalysis);
    // 1, Outline the fusion to a standalone module op.
    OwningOpRef<ModuleOp> m;
    if (failed(outlineFusionOp(fusionOp, fusionPattern, m))) {
      return fusionOp->emitError() << "failed to outlineFusionOp\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "After outline fusion op:\n" << m.get() << "\n");

    // 2, assign a default schedule for each pattern here.
    PatternDescription patternDescription(fusionOp, fusionPattern,
                                          shapeAnalysis);
    if (failed(scheduleDispatcher.dispatch(patternDescription, m.get()))) {
      return fusionOp->emitError() << "failed to assignSchedule\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "After assign schedule for fusion op:\n"
                            << m.get() << "\n");

    // 3, Build a nested pass pipeline to legalize the outlined fusion op.
    if (failed(runTransformPipeline(m.get()))) {
      return fusionOp->emitError() << "failed to run runTransformPipeline\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "After run transform pipeline:\n"
                            << m.get() << "\n");

    // 4, Inline the lowered IR into the orignal module.
    if (failed(inlineTransformedModule(b, fusion, fusionPattern, m.get()))) {
      return fusion->emitError() << "failed to inlineTransformedModule\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "After inline transformed module:\n"
                            << *fusion << "\n");
  }

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

  // Assign a transform schedule for the given fusion pattern.
  ScheduleDispatcher scheduleDispatcher{transformFileName_};
  if (failed(scheduleDispatcher.parseModuleFromFile(b.getContext()))) {
    func->emitError() << "failed to parse transform module form "
                      << transformFileName_ << " .\n";
    return signalPassFailure();
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
    if (failed(handleCpuFusionOp(b, fusion, *shapeAnalysisPtr,
                                 scheduleDispatcher))) {
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
