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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public CanonicalizerBase<Canonicalizer> {
  Canonicalizer(const SmallVectorImpl<std::string>& disabledPatterns,
                const SmallVectorImpl<std::string>& enabledPatterns) {
    for (const std::string& pattern : disabledPatterns)
      disabledPatterns_.push_back(pattern);
    for (const std::string& pattern : enabledPatterns)
      enabledPatterns_.push_back(pattern);
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet owningPatterns(context);
    for (auto* dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns_, enabledPatterns_);
    return success();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
  }

  FrozenRewritePatternSet patterns;
};
}  // end anonymous namespace

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> createDiscCanonicalizerPass(
    const SmallVector<std::string>& disabledPatterns,
    const SmallVector<std::string>& enabledPatterns) {
  return std::make_unique<Canonicalizer>(disabledPatterns, enabledPatterns);
}

}  // namespace disc_ral
}  // namespace mlir
