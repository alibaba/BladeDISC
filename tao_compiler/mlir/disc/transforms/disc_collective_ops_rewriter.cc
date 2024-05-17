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

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
enum ReductionKind {
  ALL_REDUCE_SUM,
  ALL_REDUCE_PRODUCT,
  ALL_REDUCE_MIN,
  ALL_REDUCE_MAX,
};

std::optional<std::string> ReductionKindToString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::ALL_REDUCE_SUM:
      return "sum";
    case ReductionKind::ALL_REDUCE_PRODUCT:
      return "product";
    case ReductionKind::ALL_REDUCE_MIN:
      return "min";
    case ReductionKind::ALL_REDUCE_MAX:
      return "max";
  }
  return std::nullopt;
}

std::optional<std::string> MatchReductionComputation(Region& region) {
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }

  auto ret = dyn_cast<mhlo::ReturnOp>(region.front().getTerminator());
  if (!ret || ret->getNumOperands() != 1) {
    return std::nullopt;
  }

  auto computation = ret.getOperand(0).getDefiningOp();

  if (isa<mhlo::AddOp>(computation)) {
    return "sum";
  }
  if (isa<mhlo::MulOp>(computation)) {
    return "product";
  }
  if (isa<mhlo::MinOp>(computation)) {
    return "min";
  }
  if (isa<mhlo::MaxOp>(computation)) {
    return "max";
  }
  return std::nullopt;
}

struct AllReduceOpConverter : public OpRewritePattern<mhlo::AllReduceOp> {
  using OpRewritePattern<mhlo::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Value, 4> newOutputs;
    auto reductionKind = MatchReductionComputation(op.getRegion());
    if (!reductionKind) {
      return failure();
    }
    for (int i = 0; i < op->getOperands().size(); ++i) {
      // no need call all_reduce op if no consumer
      if (op->getResult(i).getUsers().empty()) {
        continue;
      }

      op->setAttr("call_target_name", rewriter.getStringAttr("ral_all_reduce"));
      op->setAttr("device", rewriter.getStringAttr("d"));
      op->setAttr("input_placements", rewriter.getStringAttr("d"));
      op->setAttr("output_placements", rewriter.getStringAttr("d"));
      op->setAttr("input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("output_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_input_layouts", rewriter.getStringAttr("*"));
      op->setAttr("expected_output_layouts", rewriter.getStringAttr("*"));
      SmallVector<NamedAttribute> newAttrs;
      newAttrs.push_back(
          NamedAttribute(rewriter.getStringAttr("reduction_kind"),
                         rewriter.getStringAttr(reductionKind.value())));

      auto newCustomAttrs = DictionaryAttr::get(op->getContext(), newAttrs);

      op->setAttr("custom_attrs", newCustomAttrs);

      auto newOutput = rewriter.create<mhlo_disc::CustomCallV2Op>(
          op->getLoc(), op->getResults()[i].getType(), op->getOperands()[i],
          op->getAttrs());
      newOutputs.push_back(newOutput.getResult(0));
    }
    rewriter.replaceOp(op, newOutputs);
    return success();
  }
};

struct AllGatherOpConverter : public OpRewritePattern<mhlo::AllGatherOp> {
  using OpRewritePattern<mhlo::AllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AllGatherOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<NamedAttribute, 4> newAttrsVec;
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr("ral_all_gather")));
    newAttrsVec.push_back(NamedAttribute(rewriter.getStringAttr("device"),
                                         rewriter.getStringAttr("d")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("input_placements"),
                       rewriter.getStringAttr("d")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("output_placements"),
                       rewriter.getStringAttr("d")));
    newAttrsVec.push_back(NamedAttribute(
        rewriter.getStringAttr("input_layouts"), rewriter.getStringAttr("*")));
    newAttrsVec.push_back(NamedAttribute(
        rewriter.getStringAttr("output_layouts"), rewriter.getStringAttr("*")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("expected_input_layouts"),
                       rewriter.getStringAttr("*")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("expected_output_layouts"),
                       rewriter.getStringAttr("*")));

    SmallVector<NamedAttribute, 4> customAttrs;
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("all_gather_dim"),
                       op->getAttr("all_gather_dim")));
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("replica_groups"),
                       op->getAttr("replica_groups")));

    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("custom_attrs"),
                       rewriter.getDictionaryAttr(customAttrs)));

    ArrayRef<NamedAttribute> newCustomAttrs(newAttrsVec);
    auto newOp = rewriter.create<mhlo_disc::CustomCallV2Op>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), newCustomAttrs);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct ReduceScatterOpConverter
    : public OpRewritePattern<mhlo::ReduceScatterOp> {
  using OpRewritePattern<mhlo::ReduceScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceScatterOp op,
                                PatternRewriter& rewriter) const override {
    auto reductionKind = MatchReductionComputation(op.getRegion());
    if (!reductionKind) {
      return failure();
    }
    SmallVector<NamedAttribute, 4> newAttrsVec;
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr("ral_reduce_scatter")));
    newAttrsVec.push_back(NamedAttribute(rewriter.getStringAttr("device"),
                                         rewriter.getStringAttr("d")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("input_placements"),
                       rewriter.getStringAttr("d")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("output_placements"),
                       rewriter.getStringAttr("d")));
    newAttrsVec.push_back(NamedAttribute(
        rewriter.getStringAttr("input_layouts"), rewriter.getStringAttr("*")));
    newAttrsVec.push_back(NamedAttribute(
        rewriter.getStringAttr("output_layouts"), rewriter.getStringAttr("*")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("expected_input_layouts"),
                       rewriter.getStringAttr("*")));
    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("expected_output_layouts"),
                       rewriter.getStringAttr("*")));

    SmallVector<NamedAttribute, 4> customAttrs;
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("reduction_kind"),
                       rewriter.getStringAttr(reductionKind.value())));
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("scatter_dimension"),
                       op->getAttr("scatter_dimension")));
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("replica_groups"),
                       op->getAttr("replica_groups")));

    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("custom_attrs"),
                       rewriter.getDictionaryAttr(customAttrs)));

    ArrayRef<NamedAttribute> newCustomAttrs(newAttrsVec);
    auto newOp = rewriter.create<mhlo_disc::CustomCallV2Op>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), newCustomAttrs);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
}  //  namespace

struct DiscCollectiveOpsRewriterPass
    : public DiscCollectiveOpsRewriterPassBase<DiscCollectiveOpsRewriterPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<AllReduceOpConverter>(ctx);
    patterns.insert<AllGatherOpConverter>(ctx);
    patterns.insert<ReduceScatterOpConverter>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscCollectiveOpsRewriterPass() {
  return std::make_unique<DiscCollectiveOpsRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir
