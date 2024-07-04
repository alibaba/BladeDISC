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

struct LhloDISCArgsMutationOpConverter
    : public OpRewritePattern<lmhlo_disc::ArgsMutationOp> {
  explicit LhloDISCArgsMutationOpConverter(MLIRContext* context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(lmhlo_disc::ArgsMutationOp lhloOp,
                                PatternRewriter& rewriter) const override {
    auto op = lhloOp.getOperation();
    auto operands = op->getOperands();
    if (operands[0] == operands[1]) {
      rewriter.eraseOp(op);
      return success();
    }

    if (operands[0].getType().cast<MemRefType>() ==
        operands[1].getType().cast<MemRefType>()) {
      for (auto user : operands[1].getUsers()) {
        // Prevent double dealloc
        if (isa<memref::DeallocOp>(user)) {
          rewriter.eraseOp(user);
        }
      }
      operands[0].replaceAllUsesWith(operands[1]);
      rewriter.eraseOp(op);
      // rewriter.eraseOp(operands[0].getDefiningOp<memref::AllocOp>());
    } else {
      llvm::dbgs() << "Reinterprete cast need to be inserted here between "
                   << operands[0] << " and " << operands[1] << "\n";

      for (auto user : operands[1].getUsers()) {
        if (isa<memref::DeallocOp>(user)) {
          rewriter.eraseOp(user);
        }
      }

      auto shape_a = operands[0].getType().cast<MemRefType>().getShape();
      auto alloc_a = operands[0].getDefiningOp<memref::AllocOp>();
      SmallVector<Value> dimSizes;
      int dynamic_dim_idx = 0;
      for (int i = 0; i < shape_a.size(); ++i) {
        if (shape_a[i] == ShapedType::kDynamic) {
          dimSizes.push_back(alloc_a->getOperand(dynamic_dim_idx++));
        } else {
          dimSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
              op->getLoc(), shape_a[i]));
        }
      }

      Value newValue = disc_ral::CastMemRefTo(
          rewriter, op->getLoc(), operands[1],
          operands[0].getType().cast<MemRefType>(), dimSizes);
      operands[0].replaceAllUsesWith(newValue);
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct DiscArgsMutationExpandPass
    : public DiscArgsMutationExpandPassBase<DiscArgsMutationExpandPass> {
  using DiscArgsMutationExpandPassBase<
      DiscArgsMutationExpandPass>::DiscArgsMutationExpandPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect>();
  }

 public:
  DiscArgsMutationExpandPass() = default;

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<lmhlo_disc::ArgsMutationOp>();
    patterns.insert<LhloDISCArgsMutationOpConverter>(&context);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscArgsMutationExpandPass() {
  return std::make_unique<DiscArgsMutationExpandPass>();
}
}  // namespace mhlo_disc
}  // namespace mlir