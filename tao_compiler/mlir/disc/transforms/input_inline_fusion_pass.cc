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

// This file implements inline fusion.
//
#include <limits>

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/input_inline_fusion_pattern.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"

using mlir::memref::LoadOp;

namespace mlir {
namespace disc_ral {

using namespace lmhlo;

namespace {

// TODO(disc): Maybe it worth explicitly adding the I/O buffers onto the
// outlining of lmhlo::FusionOp and then mark IsolatedFromAbove for
// lmhlo::FusionOp. By this way the fusion codegen passes can be OperationPass
// on lmhlo::FusionOp for better compilation overhead.
class InputInlineFusion : public InputInlineFusionPassBase<InputInlineFusion> {
  void runOnOperation() override;
};

}  // end anonymous namespace

// This pass works after LhloLegalizeRootsToParallelLoops pass for the
// XLA-style fusion codegen.
std::unique_ptr<OperationPass<func::FuncOp>> createDiscInputInlineFusionPass() {
  return std::make_unique<InputInlineFusion>();
}

namespace {

constexpr unsigned c_MAX_ITERATION = 4096 * 1000;

void InputInlineFusion::runOnOperation() {
  func::FuncOp func = getOperation();
  auto* context = &this->getContext();
  RewritePatternSet patterns(context);
  patterns.insert<InputInlineFusionPattern>(context);

  // Just apply the patterns greedily.
  // There should always be one scf.ParallelOp in the fusion.
  auto config = GreedyRewriteConfig();
  config.maxIterations = c_MAX_ITERATION;
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
    signalPassFailure();
  }

  // there should be no lmhlo ops after inline fusion,
  // except for the ConstantOp of ColReduction, which for now cannot be
  // properly optimized by general DCE pass
  std::vector<Operation*> to_be_removed;
  func.walk([&](FusionOp fusion) {
    if (isFusionType<FusionType::kStitch>(fusion.getOperation())) return;
    fusion.getRegion().walk([&](LmhloOp op) {
      if (isa<TerminatorOp>(op)) {
        return;
      }
      if (isa<ConstantOp>(op)) {
        // TODO(disc): Check the ConstantOp is from ReduceOp
        to_be_removed.push_back(op);
        return;
      }
      if (isa<lmhlo_disc::PrintfOp>(op)) {
        return;
      }
      op.emitError("unexpected remaining operation in a FusionOp");
      signalPassFailure();
    });
  });
  for (auto op : to_be_removed) {
    op->erase();
  }
}

}  // namespace

}  // namespace disc_ral
}  // namespace mlir
