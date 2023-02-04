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

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/transforms/disc_pdl_utils.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/DiscPdlPredefinedPatterns.h"

#include "tests/torch-disc-pdll/utils.h"

using namespace mlir;
using namespace mlir::torch;

namespace {

static bool wouldOpBeTriviallyDeadImplDisc(Operation* rootOp) {
  // The set of operations to consider when checking for side effects.
  SmallVector<Operation*, 1> effectingOps(1, rootOp);
  while (!effectingOps.empty()) {
    Operation* op = effectingOps.pop_back_val();

    // If the operation has recursive effects, push all of the nested operations
    // on to the stack to consider.
    bool hasRecursiveEffects =
        op->hasTrait<::mlir::OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      // Modification 1, if has recursive effects, directly return false.
      return false;
    }

    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Check to see if this op either has no effects, or only allocates/reads
      // memory.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);

      // Gather all results of this op that are allocated.
      SmallPtrSet<Value, 4> allocResults;
      for (const MemoryEffects::EffectInstance& it : effects)
        if (isa<MemoryEffects::Allocate>(it.getEffect()) && it.getValue() &&
            it.getValue().getDefiningOp() == op)
          allocResults.insert(it.getValue());

      if (!llvm::all_of(
              effects,
              [&allocResults](const MemoryEffects::EffectInstance& it) {
                // We can drop effects if the value is an allocation and is a
                // result of the operation.
                if (allocResults.contains(it.getValue()))
                  return true;
                // Otherwise, the effect must be a read.
                return isa<MemoryEffects::Read>(it.getEffect());
              })) {
        return false;
      }
      continue;

      // Otherwise, if the op has recursive side effects we can treat the
      // operation itself as having no effects.
    }
    if (hasRecursiveEffects)
      continue;

    // If there were no effect interfaces, we treat this op as conservatively
    // having effects.
    return true;
  }

  // Modification 2:
  // If we get here, we mark the op as "dead".
  return true;
}

bool wouldOpBeTriviallyDeadDisc(Operation* op) {
  if (op->mightHaveTrait<::mlir::OpTrait::IsTerminator>()) {
    return false;
  }
  return wouldOpBeTriviallyDeadImplDisc(op);
}

bool isOpTriviallyDeadDisc(Operation* op) {
  return op->use_empty() && wouldOpBeTriviallyDeadDisc(op);
}

struct ApplyDiscPdlPatternsPass
    : public mlir::torch::TorchConversion::ApplyDiscPdlPatternsBase<
          ApplyDiscPdlPatternsPass> {
  ApplyDiscPdlPatternsPass(
      const std::string& pdll_files,
      const std::string& pdll_include_dirs)
      : ApplyDiscPdlPatternsBase<
            ApplyDiscPdlPatternsPass>::ApplyDiscPdlPatternsBase() {
    this->pdll_files_ = pdll_files;
    this->pdll_include_dirs_ = pdll_include_dirs;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mhlo_disc::MhloDiscDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<pdl_interp::PDLInterpDialect>();
    mlir::disc_ral::getPDLDependentDialects(registry);
  }
  void runOnOperation() override;
};

struct PdlDeadCodeElimination : public RewritePattern {
  PdlDeadCodeElimination(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  bool isInplaceSafe(Value& input) const {
    if (input.getType().isa<mlir::torch::Torch::NonValueTensorType>()) {
      return false;
    }
    return true;
  }

  LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter)
      const override {
    for (Value operand : op->getOperands()) {
      // All inputs must not be NonValueTensorType.
      if (!isInplaceSafe(operand)) {
        return failure();
      }
    }
    if (!isOpTriviallyDeadDisc(op)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void ApplyDiscPdlPatternsPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  auto pdll_include_dirs = mlir::disc_ral::ParseFileString(pdll_include_dirs_);
  (void)mlir::disc_ral::populateDiscPdlPatternsFromString(
      &patterns,
      getTorchPredefinedPDLPatterns(),
      pdll_include_dirs,
      torch::kDefaultHelperFunctionDeclarations,
      torch::registerPredefinedHelperFunctions);

  (void)mlir::disc_ral::populateDiscPdlPatternsFromFiles(
      &patterns,
      mlir::disc_ral::ParseFileString(pdll_files_),
      pdll_include_dirs,
      torch::kDefaultHelperFunctionDeclarations,
      torch::registerPredefinedHelperFunctions);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  // All ops in torch-mlir have side effects, so that the dce pass
  // in mlir cannot take effect on the graph of torch-mlir.
  // So we copied the code of dce from mlir and made some modifications,
  // so that it can take effect on the graph of torch-mlir.
  RewritePatternSet pdlDecPatterns(context);
  pdlDecPatterns.add<PdlDeadCodeElimination>(context);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(pdlDecPatterns))))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createApplyDiscPdlPatternsPass(
        const std::string& pdll_files,
        const std::string& pdll_include_dirs) {
  return std::make_unique<ApplyDiscPdlPatternsPass>(
      pdll_files, pdll_include_dirs);
}
