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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
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

struct LhloDiscEvictOpConverter
    : public OpRewritePattern<lmhlo_disc::EvictOp> {
  explicit LhloDiscEvictOpConverter(MLIRContext* context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(lmhlo_disc::EvictOp lhloOp,
                                PatternRewriter& rewriter) const override {
    
    auto op = lhloOp.getOperation();
    SmallVector<Value, 4> newOutputs;
    for (int i = 0; i < op->getOperands().size(); ++i) {
      op->setAttr("call_target_name", rewriter.getStringAttr("ral_evict"));
      op->setAttr("device", rewriter.getStringAttr("d"));
      op->setAttr("input_placements", rewriter.getStringAttr("d"));
      op->setAttr("output_placements", rewriter.getStringAttr("d"));
      op->setAttr("input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("output_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_output_layouts", rewriter.getStringAttr("*"));

      SmallVector<NamedAttribute> attrs;
      auto customAttrs = DictionaryAttr::get(op->getContext(), attrs);
      op->setAttr("custom_attrs", customAttrs);

      auto evict_op = rewriter.create<lmhlo_disc::CustomCallV2Op>(
          op->getLoc(), op->getResults()[i].getType(), op->getOperands()[i],
          op->getAttrs());
      newOutputs.push_back(evict_op.getResult(0));
    }
    rewriter.replaceOp(op, newOutputs);
    return success();
  }
};


struct LhloDiscEvictionCheckOpConverter
    : public OpRewritePattern<lmhlo_disc::EvictionCheckOp> {
  explicit LhloDiscEvictionCheckOpConverter(MLIRContext* context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(lmhlo_disc::EvictionCheckOp lhloOp,
                                PatternRewriter& rewriter) const override {

    auto op = lhloOp.getOperation();
    SmallVector<Value, 4> newOutputs;
    for (int i = 0; i < op->getOperands().size(); ++i) {
      op->setAttr("call_target_name", rewriter.getStringAttr("ral_eviction_check"));
      op->setAttr("device", rewriter.getStringAttr("d"));
      op->setAttr("input_placements", rewriter.getStringAttr("d"));
      op->setAttr("output_placements", rewriter.getStringAttr("h"));
      op->setAttr("input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("output_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_output_layouts", rewriter.getStringAttr("*"));

      
      SmallVector<NamedAttribute> attrs;
      auto customAttrs = DictionaryAttr::get(op->getContext(), attrs);
      op->setAttr("custom_attrs", customAttrs);

      auto evict_op = rewriter.create<lmhlo_disc::CustomCallV2Op>(
          op->getLoc(), op->getResults()[i].getType(), op->getOperands()[i],
          op->getAttrs());
      newOutputs.push_back(evict_op.getResult(0));
    }
    rewriter.replaceOp(op, newOutputs);
    return success();
  }
};


struct DiscEvictExpandPass
    : public DiscEvictExpandPassBase<DiscEvictExpandPass> {
  using DiscEvictExpandPassBase<
      DiscEvictExpandPass>::DiscEvictExpandPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect>();
  }

 public:
  DiscEvictExpandPass() = default;

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<lmhlo_disc::EvictOp>();
    target.addIllegalOp<lmhlo_disc::EvictionCheckOp>();

    patterns.insert<LhloDiscEvictOpConverter>(&context);
    patterns.insert<LhloDiscEvictionCheckOpConverter>(&context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscEvictExpandPass() {
  return std::make_unique<DiscEvictExpandPass>();
}
}  // namespace mhlo_disc
}  // namespace mlir