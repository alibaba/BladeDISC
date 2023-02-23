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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_pdl_utils.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "disc-sparse-op-rewriter"

namespace mlir {
namespace disc_ral {

// sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
//     math_ops.reduce_prod(
//         array_ops.slice(original_shape, [0], [original_rank - 1])),
//     array_ops.gather(original_shape, original_rank - 1)
// ])
// For 2d sparse tensor, as the above logic from
// tf.safe_embedding_lookup_sparse, sparse_reshape can be eliminated
// sparse-reshape pdll patterns here.
std::string getSparseReshapePDLPattern() {
  std::string preDefinedPatterns;
  preDefinedPatterns += R"pdll(
    Pattern SparseReshapeSimplifyPattern {
      /// match phase: define the pattern
      let c0 = op<arith.constant> {value = attr<"0 : index">} -> (type<"index">);
      let c0_i64 = op<arith.constant> {value = attr<"0 : i64">} -> (type<"i64">);
      let c1_i64 = op<arith.constant> {value = attr<"1 : i64">} -> (type<"i64">);
      let dim = op<tensor.dim>(
          arg24: Value,
          c0
      );
      let cast = op<arith.index_cast>(
          dim.0
      );
      let sub = op<arith.subi>(
          cast.0,
          c1_i64
      );

      let min = op<arith.minsi>(
          sub.0,
          c0_i64
      );
      let add = op<arith.addi>(
          min.0,
          c1_i64
      );
      let from_ele = op<tensor.from_elements>(
          min.0
      );
      let from_ele_1 = op<tensor.from_elements>(
          add.0
      );
      let slice = op<mhlo.real_dynamic_slice>(
          arg24,
          from_ele.0,
          from_ele_1.0,
          cst: Value
      );
      let reduce = op<mhlo.reduce>(
          slice.0,
          init: Value
      );
      let reshape = op<mhlo.reshape>(
          reduce.0
      );
      //
      let gather = op<mhlo.dynamic_gather>(
          arg24,
          cst_5: Value,
          cst_1: Value
      );
      let reshape_1 = op<mhlo.reshape>(
          gather.0
      );

      let concat = op<mhlo.concatenate>(
          reshape.0,
          reshape_1.0
      );
      // sparse_reshape
      let sparse_reshape = op<mhlo_disc.sparse_reshape>(
          arg86: Value,
          arg24,
          concat.0
      );
      /// check constant tensor constant
      CheckConstantTensorValueIs(cst, attr<"1">);
      CheckConstantTensorValueIs(cst_1, attr<"1">);
      CheckConstantTensorValueIs(init, attr<"1">);
      CheckConstantTensorValueIs(cst_5, attr<"1">);

      // check rank-2 input?
      // rewrite phase
      rewrite sparse_reshape with {
          // Just eliminate sparse_reshape
          let rs = PackValue_2(attr<"\"out\"">, arg86, arg24);
          replace sparse_reshape with rs;
      };
    }
  )pdll";
  return preDefinedPatterns;
}

struct DiscSparseOpRewriterPass
    : public DiscSparseOpRewriterPassBase<DiscSparseOpRewriterPass> {
  void runOnOperation() override;
};

void DiscSparseOpRewriterPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  // sparse reshape pattern
  populateDiscPdlPatternsFromString(&patterns, getSparseReshapePDLPattern());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createDiscSparseOpRewriterPass() {
  return std::make_unique<DiscSparseOpRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir
