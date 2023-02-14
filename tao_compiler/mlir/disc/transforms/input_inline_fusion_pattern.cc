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

#include "mlir/disc/transforms/input_inline_fusion_pattern.h"

#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/codegen_utils.h"

namespace mlir {
namespace disc_ral {

using namespace lmhlo;

template <typename LHLO_OpTy>
bool elemwiseFuseHelper(PatternRewriter& rewriter, Operation* user,
                        Operation* producer, memref::LoadOp load_op,
                        const SmallVector<memref::LoadOp>& load_ops,
                        LowerConfig* lower_config) {
  if (!isa<LHLO_OpTy>(producer) ||
      !LHLO_OpTy::template hasTrait<OpTrait::Elementwise>())
    return false;
  auto loc = user->getLoc();
  SmallVector<Value, 4> operand_values;
  unsigned num_operands = producer->getNumOperands();
  for (unsigned i = 0; i < num_operands - 1; ++i) {
    auto producer_operand = producer->getOperand(i);
    rewriter.setInsertionPoint(load_op);
    operand_values.push_back(
        createMaySpecificLoad(rewriter, loc, producer, producer_operand,
                              load_op.getIndices(), lower_config));
  }
  auto inlined_result =
      LhloOpToStdScalarOp::map<LHLO_OpTy>(llvm::cast<LHLO_OpTy>(producer),
                                          cast<lmhlo::LmhloOp>(producer)
                                              .getResultBuffer()
                                              .getType()
                                              .cast<MemRefType>()
                                              .getElementType(),
                                          operand_values, &rewriter);

  for (memref::LoadOp to_be_replaced : load_ops)
    to_be_replaced.replaceAllUsesWith(inlined_result);
  mayCreateStore(&rewriter, loc, producer, inlined_result, load_op.getIndices(),
                 lower_config);
  return true;
}

template <typename LHLO_OpTy>
bool miscFuseHelper(PatternRewriter& rewriter, Operation* user,
                    Operation* opaque_producer, memref::LoadOp load_op,
                    const SmallVector<memref::LoadOp>& load_ops,
                    LowerConfig* lower_config) {
  LHLO_OpTy producer = dyn_cast<LHLO_OpTy>(opaque_producer);
  if (!producer) return false;
  auto loc = user->getLoc();
  rewriter.setInsertionPoint(load_op);
  auto inlined_result =
      elementalLower<LHLO_OpTy>(&rewriter, loc, producer, load_op.getIndices(),
                                /* check cache */ true, lower_config);
  for (memref::LoadOp to_be_replaced : load_ops) {
    to_be_replaced.replaceAllUsesWith(inlined_result);
  }
  return true;
}

template <>
bool miscFuseHelper<ConstantOp>(PatternRewriter& rewriter, Operation* user,
                                Operation* producer, memref::LoadOp load_op,
                                const SmallVector<memref::LoadOp>& load_ops,
                                LowerConfig* lower_config) {
  if (!isa<ConstantOp>(producer)) return false;
  auto constant = cast<lmhlo::ConstantOp>(producer);
  auto memref_type = constant.getOutput().getType().cast<MemRefType>();
  bool is_splat = constant.getValue().isSplat();
  assert((memref_type.getRank() == 0 || is_splat) &&
         "only scalar ConstantOp can be fused");
  auto loc = user->getLoc();
  rewriter.setInsertionPoint(load_op);
  auto elem_ty = memref_type.getElementType();
  bool need_cast = elem_ty.isUnsignedInteger() || elem_ty.isSignedInteger();
  Value inlined_result = nullptr;
  if (need_cast) {
    int64_t val =
        is_splat ? constant.getValue().getSplatValue<APInt>().getSExtValue()
                 : constant.getValue().getValues<APInt>()[{}].getSExtValue();
    Value const_val = rewriter.create<arith::ConstantOp>(
        loc,
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(elem_ty.cast<IntegerType>().getWidth()),
            val));

    SmallVector<Type> elem_types(1, elem_ty);
    inlined_result =
        rewriter.create<UnrealizedConversionCastOp>(loc, elem_types, const_val)
            .getResults()
            .front();
  } else {
    auto attr = is_splat ? constant.getValue().getSplatValue<Attribute>()
                         : constant.getValue().getValues<Attribute>()[{}];
    inlined_result = rewriter.create<arith::ConstantOp>(loc, elem_ty, attr);
  }
  for (memref::LoadOp to_be_replaced : load_ops)
    to_be_replaced.replaceAllUsesWith(inlined_result);
  return true;
}

template <typename First>
bool elemwiseFuseHelperOr(PatternRewriter& rewriter, Operation* user,
                          Operation* producer, memref::LoadOp load_op,
                          const SmallVector<memref::LoadOp>& load_ops,
                          LowerConfig* lower_config) {
  return elemwiseFuseHelper<First>(rewriter, user, producer, load_op, load_ops,
                                   lower_config);
}

template <typename First, typename Second, typename... Rest>
bool elemwiseFuseHelperOr(PatternRewriter& rewriter, Operation* user,
                          Operation* producer, memref::LoadOp load_op,
                          const SmallVector<memref::LoadOp>& load_ops,
                          LowerConfig* lower_config) {
  return elemwiseFuseHelperOr<First>(rewriter, user, producer, load_op,
                                     load_ops, lower_config) ||
         elemwiseFuseHelperOr<Second, Rest...>(rewriter, user, producer,
                                               load_op, load_ops, lower_config);
}

LogicalResult InputInlineFusionPattern::processParallelOp(
    scf::ParallelOp parallel_op, Block* parent_block, PatternRewriter& rewriter,
    const DominanceInfo& dominance_info) const {
  SmallVector<memref::LoadOp, 4> load_ops;
  parallel_op->walk(
      [&](memref::LoadOp load_op) { load_ops.push_back(load_op); });
  for (auto load_op : load_ops) {
    auto lhlo_op = getFusibleOperation(load_op);
    if (!lhlo_op) continue;
    // 1, in case of:
    //      A = ...
    //      B = op(A)
    //      C = op(A, B)
    //    C should fuse B first before fusing A.
    //    This is the same logic as in instruction_fusion pass of XLA
    //
    // 2, When multiple loads consume the same result of lhlo_op and
    //    the load indices are also identical, the ir should be
    //    emitted only once. Other LoadOps should use cached Value.

    // 'load_ops' that can consume the same cached value
    SmallVector<memref::LoadOp> same_load_ops;
    bool can_remove_producer;
    if (!checkIfFusible(parallel_op, lhlo_op, load_op, can_remove_producer,
                        same_load_ops, dominance_info))
      continue;
    // 'load_op' is always the one that locates in the most
    // external code block among all the 'same_load_ops', because the walker
    // is in the post order sequence.
    if (failed(inlineFuseLhloOp(rewriter, parallel_op, lhlo_op, load_op,
                                same_load_ops, lower_config_)))
      return failure();
    if (can_remove_producer) rewriter.eraseOp(lhlo_op);
    for (memref::LoadOp to_be_removed : same_load_ops)
      rewriter.eraseOp(to_be_removed);

    // Clean all the ops that do not have LoadOps inside the nested
    // ParallelOps and is not the ancestor of any ops that have LoadOps
    // inside the nested ParallelOps.
    cleanUnusedLhloOps(parent_block, &rewriter);
    return success();
  }
  return failure();
}

Operation* InputInlineFusionPattern::getFusibleOperation(
    memref::LoadOp load_op) const {
  Operation* lhlo_op = nullptr;
  for (auto* user : getValueUsersInFusionLike(load_op.getMemRef(), load_op)) {
    if (isa<LmhloOp>(user) &&
        isSameUnderlineBuffer(cast<LmhloOp>(user).getResultBuffer(),
                              load_op.getOperation()->getOperand(0))) {
      if (lhlo_op)
        llvm::report_fatal_error(
            "More than one lhlo_op write to one Memref within one fusion");
      lhlo_op = user;
    }
  }
  return lhlo_op;
}

// Check if there are no other consumers of the producer
// except the ParallelOp.
bool InputInlineFusionPattern::checkIfFusible(
    scf::ParallelOp user, Operation* producer, memref::LoadOp load_op,
    bool& can_remove_producer, SmallVector<memref::LoadOp>& load_ops,
    const DominanceInfo& dominance_info) const {
  load_ops.clear();
  assert(isa<LmhloOp>(producer) && "Unexpected producer in checkIfFusible");
  auto producer_result_memref = cast<LmhloOp>(producer).getResultBuffer();
  can_remove_producer = true;
  auto lhlo_dialect = user->getContext()->getLoadedDialect("lmhlo");
  for (auto* memref_user :
       getValueUsersInFusionLike(producer_result_memref, producer)) {
    if ((memref_user->getDialect() == lhlo_dialect) &&
        (memref_user != producer)) {
      return false;
    }
    memref::LoadOp other = dyn_cast<memref::LoadOp>(memref_user);
    if (!other) continue;
    if (other.getMemRef() == load_op.getMemRef() &&
        other.getIndices() == load_op.getIndices() &&
        dominance_info.dominates(load_op.getOperation(),
                                 other.getOperation())) {
      // Since load_op -> other is in the forward walk order, so it's only
      // possible for load_op to dominate other.
      // TODO: If load_op and other cannot dominate each other,
      // still it's possible to find a dominant insertion point and emit the
      // fused IRs. However, More works are needed to analyse at compile time if
      // it's win to do this.
      load_ops.emplace_back(other);
    } else {
      can_remove_producer = false;
    }
  }
  return true;
}

// load_op is among the load_ops, whose locates in the most
// external code block
LogicalResult InputInlineFusionPattern::inlineFuseLhloOp(
    PatternRewriter& b, Operation* user, Operation* producer,
    memref::LoadOp load_op, const SmallVector<memref::LoadOp>& load_ops,
    LowerConfig* lower_config) const {
  if (elemwiseFuseHelperOr<
#define GET_SUPPORTED_OP_LIST
#include "mlir/disc/transforms/disc_supported_list.h.inc"
          >(b, user, producer, load_op, load_ops, lower_config) ||
      // TODO(disc): Upstream is on the way for more Ops
      miscFuseHelper<BroadcastInDimOp>(b, user, producer, load_op, load_ops,
                                       lower_config) ||
      miscFuseHelper<BroadcastOp>(b, user, producer, load_op, load_ops,
                                  lower_config) ||
      miscFuseHelper<ClampOp>(b, user, producer, load_op, load_ops,
                              lower_config) ||
      miscFuseHelper<ConcatenateOp>(b, user, producer, load_op, load_ops,
                                    lower_config) ||
      miscFuseHelper<ConstantOp>(b, user, producer, load_op, load_ops,
                                 lower_config) ||
      miscFuseHelper<CopyOp>(b, user, producer, load_op, load_ops,
                             lower_config) ||
      miscFuseHelper<DynamicBroadcastInDimOp>(b, user, producer, load_op,
                                              load_ops, lower_config) ||
      miscFuseHelper<DynamicGatherOp>(b, user, producer, load_op, load_ops,
                                      lower_config) ||
      miscFuseHelper<lmhlo_disc::H2DOp>(b, user, producer, load_op, load_ops,
                                        lower_config) ||
      miscFuseHelper<DynamicIotaOp>(b, user, producer, load_op, load_ops,
                                    lower_config) ||
      miscFuseHelper<DynamicPadOp>(b, user, producer, load_op, load_ops,
                                   lower_config) ||
      miscFuseHelper<DynamicReshapeOp>(b, user, producer, load_op, load_ops,
                                       lower_config) ||
      miscFuseHelper<GatherOp>(b, user, producer, load_op, load_ops,
                               lower_config) ||
      miscFuseHelper<IotaOp>(b, user, producer, load_op, load_ops,
                             lower_config) ||
      miscFuseHelper<IsFiniteOp>(b, user, producer, load_op, load_ops,
                                 lower_config) ||
      miscFuseHelper<NotOp>(b, user, producer, load_op, load_ops,
                            lower_config) ||
      miscFuseHelper<RealDynamicSliceOp>(b, user, producer, load_op, load_ops,
                                         lower_config) ||
      miscFuseHelper<ReduceOp>(b, user, producer, load_op, load_ops,
                               lower_config) ||
      miscFuseHelper<ReshapeOp>(b, user, producer, load_op, load_ops,
                                lower_config) ||
      miscFuseHelper<ReverseOp>(b, user, producer, load_op, load_ops,
                                lower_config) ||
      miscFuseHelper<SliceOp>(b, user, producer, load_op, load_ops,
                              lower_config) ||
      // miscFuseHelper<DynamicUpdateSliceOp>(b, user, producer, load_op,
      // load_ops, lower_config) ||
      miscFuseHelper<TransposeOp>(b, user, producer, load_op, load_ops,
                                  lower_config)) {
    return success();
  }

  return failure();
}

}  // namespace disc_ral
}  // namespace mlir
