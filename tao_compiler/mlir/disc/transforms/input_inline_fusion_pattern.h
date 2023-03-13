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

#ifndef DISC_TRANSFORMS_INPUT_INLINE_FUSION_PATTERN_H_
#define DISC_TRANSFORMS_INPUT_INLINE_FUSION_PATTERN_H_

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"

namespace mlir {
namespace disc_ral {

// This pattern iteratively looks for the lmhlo op which is the direct producer
// of the nested loops, and then inline fuse it if the fusion will not form a
// cycle.
//
// The inline fusion action can be generalized as:
// step 1: replace the producer Lhlo op into associate std op inside the nested
// loops. step 2: remove the original Load ops inside the loops and insert new
// Load ops.
//
// If there are multiple LoadOps with the same indices, they will be replaced
// with the same op. This obtains the similar result as GeneratedValueCache.
//
// IR after LhloLegalizeRootsToParallelLoops:
//    "lmhlo.fusion"() ( {
//       lmhlo.aaa(%0, %1, %2)
//       lmhlo.bbb(%2, %3, %4)
//       scf.parallel (...) {
//          memref.load %4[...]
//          ...
//          memref.store ...
//       }
//    })
//
// IR after one round of InputInlineFusionPattern:
//    "lmhlo.fusion"() ( {
//       lmhlo.aaa(%0, %1, %2)
//       scf.parallel (...) {
//          memref.load %2[...]
//          ...
//          memref.store ...
//       }
//    })
//
// Final IR after this pass:
//    "lmhlo.fusion"() ( {
//       scf.parallel (...) {
//          memref.load ...
//          ...
//          memref.store ...
//       }
//    })
class InputInlineFusionPattern : public RewritePattern {
 public:
  explicit InputInlineFusionPattern(MLIRContext* context,
                                    LowerConfig* lower_config = nullptr,
                                    bool one_pass = false)
      : RewritePattern(lmhlo::FusionOp::getOperationName(), 1, context),
        lower_config_(lower_config) {}

  LogicalResult processParallelOp(scf::ParallelOp parallel_op,
                                  Block* parent_block,
                                  PatternRewriter& rewriter,
                                  const DominanceInfo& dominance_info) const;

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (isFusionType<FusionType::kStitch>(op) && !isOnGpu(op)) return failure();
    // When we pass lower_config, we only process kStitch fusion on GPU.
    if (lower_config_ != nullptr) {
      if (!isOnGpu(op) || !isFusionType<FusionType::kStitch>(op)) {
        return failure();
      }
    }
    // skip if not the most outter ParallelOp
    auto fusion = cast<lmhlo::FusionOp>(op);
    auto& parent_block = fusion.getRegion().front();
    DominanceInfo dominance_info(op);

    SmallVector<scf::ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(op, innermostPloops);

    // Returns success if any of parallelOp is processed.
    for (scf::ParallelOp parallelOp : innermostPloops) {
      if (!failed(processParallelOp(parallelOp, &parent_block, rewriter,
                                    dominance_info))) {
        return success();
      }
    }
    return failure();
  }

 private:
  Operation* getFusibleOperation(memref::LoadOp load_op) const;
  LogicalResult inlineFuseLhloOp(PatternRewriter& b, Operation* user,
                                 Operation* producer, memref::LoadOp load_op,
                                 const SmallVector<memref::LoadOp>& load_ops,
                                 LowerConfig* lower_config) const;
  bool checkIfFusible(scf::ParallelOp user, Operation* producer,
                      memref::LoadOp load_op, bool& can_remove_producer,
                      SmallVector<memref::LoadOp>& load_ops,
                      const DominanceInfo& dominance_info) const;

  LowerConfig* lower_config_;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TRANSFORMS_INPUT_INLINE_FUSION_PATTERN_H_
