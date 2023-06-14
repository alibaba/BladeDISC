// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements logic for lowering HLO DISC dialect to LHLO DISC
// dialect.

#include <algorithm>
#include <utility>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "mlir/disc/transforms/rewriters.h"
#include "mlir/disc/transforms/shape_utils.h"

namespace mlir {
using placement_utils::kDiscPlaceAssignment;
using placement_utils::kGpu;

namespace mhlo_disc {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

struct LhloConcatenateOpConverter
    : public OpRewritePattern<lmhlo::ConcatenateOp> {
  explicit LhloConcatenateOpConverter(MLIRContext* context)
      : OpRewritePattern(context) {}

  bool isFixedShape(lmhlo::ConcatenateOp op) const {
    int operands = op.getNumOperands();
    disc_ral::ShapeConstraintIRAnalysis shape_analysis(op.getOperation());
    bool is_shape_equal = true;
    // check all input operand, the last one is output buffer
    for (int i = 0; i < operands - 2; ++i) {
      is_shape_equal &=
          shape_analysis.isShapeEqual(op->getOperand(i), op->getOperand(i + 1));
    }
    return is_shape_equal;
  }

  LogicalResult matchAndRewrite(lmhlo::ConcatenateOp lhloOp,
                                PatternRewriter& rewriter) const override {
    Operation* op = lhloOp.getOperation();
    if (!isFixedShape(lhloOp)) return failure();

    auto operands = op->getOperands();

    // TODO(yancey): support CPU place
    if (!placement_utils::isGpuMemRef(operands[0])) return failure();
    int num_input_operands = op->getNumOperands() - 1;

    SmallVector<Value, 4> ptr_array;
    auto ptr_type = IntegerType::get(op->getContext(), 64);
    SmallVector<Value, 4> ins(operands.begin(), operands.end());
    auto out = ins.back();
    auto ptr_alloc =
        rewriter
            .create<memref::AllocOp>(
                op->getLoc(), MemRefType::get({operands.size()}, ptr_type))
            .getResult();
    for (int i = 0; i < num_input_operands; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
      auto ptr = rewriter.create<disc_ral::GetPointerOp>(op->getLoc(), ptr_type,
                                                         op->getOperand(i));
      ptr->setAttr(kDiscPlaceAssignment, rewriter.getStringAttr(kGpu));
      rewriter.create<memref::StoreOp>(op->getLoc(), ptr, ptr_alloc, idx);
    }
    auto device_ptr_alloc =
        rewriter
            .create<memref::AllocOp>(
                op->getLoc(), MemRefType::get({operands.size()}, ptr_type))
            .getResult();
    rewriter.create<lmhlo_disc::H2DOp>(op->getLoc(), ptr_alloc,
                                       device_ptr_alloc);
    // {inputs, input_ptr, out}
    ins.insert(ins.begin() + num_input_operands, device_ptr_alloc);
    rewriter.create<lmhlo_disc::ConcatenateOp>(op->getLoc(), llvm::None, ins,
                                               op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DiscLhloRewriterPass
    : public DiscLhloRewriterPassBase<DiscLhloRewriterPass> {
  using DiscLhloRewriterPassBase<
      DiscLhloRewriterPass>::DiscLhloRewriterPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect,
                    disc_ral::RalDialect, lmhlo::LmhloDialect>();
  }

 public:
  DiscLhloRewriterPass() = default;

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<lmhlo::ConcatenateOp>();

    patterns.insert<LhloConcatenateOpConverter>(&context);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscLhloRewriterPass() {
  return std::make_unique<DiscLhloRewriterPass>();
}

}  // namespace mhlo_disc
}  // namespace mlir
