/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file canonicalize conv ops in hlo dialect to match the
// format of CUDNN library call.
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"      // TF:llvm-project
#include "mlir/IR/Operation.h"       // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

// The basic logic of this pass is to insert transpose op to ensure the conv op
// having expected format.
struct DiscQuantizedDotRewriterPass
    : public QuantizedDotRewriterPassBase<DiscQuantizedDotRewriterPass> {
  explicit DiscQuantizedDotRewriterPass()
      : QuantizedDotRewriterPassBase<
            DiscQuantizedDotRewriterPass>::QuantizedDotRewriterPassBase() {}

  RankedTensorType GetTransposeOutputType(
      Value value, const SmallVectorImpl<int64_t>& transpose_permutation,
      OpBuilder& b) {
    // Compute the resulting shape.
    llvm::SmallVector<int64_t, 4> transposed_shape;
    ShapedType input_type = value.getType().cast<ShapedType>();
    auto input_shape = input_type.getShape();
    for (int64_t val : transpose_permutation) {
      transposed_shape.push_back(input_shape[val]);
    }
    return RankedTensorType::get(transposed_shape, input_type.getElementType());
  }

  Operation* InsertTranspose(
      mlir::mhlo_disc::QuantizedDotGeneralOp op, Value value,
      const SmallVectorImpl<int64_t>& transpose_permutation, OpBuilder& b) {
    auto transpose_permutation_attr =
        GetI64ElementsAttr(transpose_permutation, &b);

    auto transpose_type =
        GetTransposeOutputType(value, transpose_permutation, b);
    auto transpose_op = b.create<mhlo::TransposeOp>(
        op.getLoc(), transpose_type, value, transpose_permutation_attr);

    if (auto attr = op->getAttr(placement_utils::kDiscPlaceAssignment))
      transpose_op->setAttr(placement_utils::kDiscPlaceAssignment, attr);

    return transpose_op;
  }

  LogicalResult RewriteOp(mhlo_disc::QuantizedDotGeneralOp op) {
    bool onGpu = placement_utils::isGpuMhlo(op);
    // Only need to transpose weight on gpu currently
    if (onGpu == false) {
      return success();
    }

    SmallVector<int64_t> transposeAttr = {1, 0};

    OpBuilder b(op);
    auto transpose_op = InsertTranspose(op, op.weight(), transposeAttr, b);
    op.getOperation()->setOperand(1, transpose_op->getResult(0));

    return success();
  }

  void runOnOperation() override {
    SmallVector<mhlo_disc::QuantizedDotGeneralOp, 2> ops;
    this->getOperation().walk(
        [&](mhlo_disc::QuantizedDotGeneralOp op) { ops.push_back(op); });

    for (auto& op : ops) {
      if (failed(RewriteOp(op))) {
        this->signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscQuantizedDotRewriter() {
  return std::make_unique<DiscQuantizedDotRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir
