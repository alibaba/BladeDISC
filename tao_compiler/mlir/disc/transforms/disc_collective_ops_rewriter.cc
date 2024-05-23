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

bool EnableAsyncCollective(Operation* op) {
  if (const char* env_p = std::getenv("ENABLE_ASYNC_COLLECTIVE")) {
    return std::strcmp(env_p, "true") == 0 || std::strcmp(env_p, "True") == 0;
  }
  return false;
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

    bool enable_async = EnableAsyncCollective(op.getOperation());

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

      SmallVector<NamedAttribute> attrs;
      attrs.push_back(
          NamedAttribute(rewriter.getStringAttr("reduction_kind"),
                         rewriter.getStringAttr(reductionKind.value())));
      attrs.push_back(NamedAttribute(rewriter.getStringAttr("is_async"),
                                     rewriter.getBoolAttr(enable_async)));

      auto customAttrs = DictionaryAttr::get(op->getContext(), attrs);

      op->setAttr("custom_attrs", customAttrs);

      auto reduce_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
          op->getLoc(), op->getResults()[i].getType(), op->getOperands()[i],
          op->getAttrs());

      if (enable_async) {
        int64_t async_pair_token =
            reinterpret_cast<int64_t>(reduce_op.getOperation());
        attrs.push_back(
            NamedAttribute(rewriter.getStringAttr("async_token_key"),
                           rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                                   async_pair_token)));
        auto newCustomAttrs =
            DictionaryAttr::get(reduce_op->getContext(), attrs);
        reduce_op->setAttr("custom_attrs", newCustomAttrs);

        auto original_consumer = *(op->getResults()[i].user_begin());

        // Insert CollectiveDoneOp
        auto collective_done_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
            reduce_op->getLoc(), reduce_op->getResults()[0].getType(),
            reduce_op->getResults()[0], reduce_op->getAttrs());
        collective_done_op->setAttr(
            "call_target_name",
            rewriter.getStringAttr("ral_async_collective_done"));
        newOutputs.push_back(collective_done_op.getResult(0));
      } else {
        newOutputs.push_back(reduce_op.getResult(0));
      }
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

    bool enable_async = EnableAsyncCollective(op.getOperation());

    SmallVector<NamedAttribute, 4> customAttrs;
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("all_gather_dim"),
                       op->getAttr("all_gather_dim")));
    customAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("replica_groups"),
                       op->getAttr("replica_groups")));
    customAttrs.push_back(NamedAttribute(rewriter.getStringAttr("is_async"),
                                         rewriter.getBoolAttr(enable_async)));

    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("custom_attrs"),
                       rewriter.getDictionaryAttr(customAttrs)));

    ArrayRef<NamedAttribute> allGatherOpAttrs(newAttrsVec);
    auto all_gather_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        allGatherOpAttrs);

    if (enable_async) {
      int64_t async_pair_token =
          reinterpret_cast<int64_t>(all_gather_op.getOperation());
      customAttrs.push_back(
          NamedAttribute(rewriter.getStringAttr("async_token_key"),
                         rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                                 async_pair_token)));
      auto newCustomAttrs =
          DictionaryAttr::get(all_gather_op->getContext(), customAttrs);
      all_gather_op->setAttr("custom_attrs", newCustomAttrs);

      auto original_consumer = *(op->getResult(0).user_begin());

      // Insert CollectiveDoneOp
      auto collective_done_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
          all_gather_op->getLoc(), all_gather_op->getResult(0).getType(),
          all_gather_op->getResult(0), all_gather_op->getAttrs());
      collective_done_op->setAttr(
          "call_target_name",
          rewriter.getStringAttr("ral_async_collective_done"));
      rewriter.replaceOp(op, collective_done_op.getResult(0));
    } else {
      rewriter.replaceOp(op, all_gather_op.getResult(0));
    }

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

    bool enable_async = EnableAsyncCollective(op.getOperation());

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
    customAttrs.push_back(NamedAttribute(rewriter.getStringAttr("is_async"),
                                         rewriter.getBoolAttr(enable_async)));

    newAttrsVec.push_back(
        NamedAttribute(rewriter.getStringAttr("custom_attrs"),
                       rewriter.getDictionaryAttr(customAttrs)));

    ArrayRef<NamedAttribute> reduceScatterOpAttrs(newAttrsVec);
    auto reduce_scatter_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        reduceScatterOpAttrs);

    if (enable_async) {
      int64_t async_pair_token =
          reinterpret_cast<int64_t>(reduce_scatter_op.getOperation());
      customAttrs.push_back(
          NamedAttribute(rewriter.getStringAttr("async_token_key"),
                         rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                                 async_pair_token)));
      auto newCustomAttrs =
          DictionaryAttr::get(reduce_scatter_op->getContext(), customAttrs);
      reduce_scatter_op->setAttr("custom_attrs", newCustomAttrs);

      auto original_consumer = *(op->getResult(0).user_begin());
      // Insert CollectiveDoneOp
      auto collective_done_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
          reduce_scatter_op->getLoc(),
          reduce_scatter_op->getResult(0).getType(),
          reduce_scatter_op->getResult(0), reduce_scatter_op->getAttrs());
      collective_done_op->setAttr(
          "call_target_name",
          rewriter.getStringAttr("ral_async_collective_done"));
      rewriter.replaceOp(op, collective_done_op.getResult(0));
    } else {
      rewriter.replaceOp(op, reduce_scatter_op.getResult(0));
    }
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
