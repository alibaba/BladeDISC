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

#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/AllocTensorElimination.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/utils.h"
#include "mlir/include/mlir/Dialect/Utils/IndexingUtils.h"

namespace mlir {
namespace disc_ral {
namespace transform_dialect {

CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.cc.inc"
      >();
}

//===---------------------------------------------------------------------===//
// DISCBufferizeOp
//===---------------------------------------------------------------------===//

using bufferization::BufferizationOptions;
using bufferization::OneShotAnalysisState;
using bufferization::OneShotBufferizationOptions;
using disc_linalg_ext::ConstantWrapperOp;
using disc_linalg_ext::MultiLevelPackOp;

namespace {

bool betterToUseAlloca(ShapedType type) {
  constexpr unsigned kMaximumSizeInBytes = 64 * 1024;  // 64kB
  constexpr unsigned kBitwidthOfIndexType = 64;

  if (!type || !type.hasStaticShape()) return false;

  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? kBitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> comprehenciveBufferizeAllocationFn(
    OpBuilder& builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocOp>(loc, memRefType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

LogicalResult comprehensiveBufferizeDeallocationFn(OpBuilder& builder,
                                                   Location loc,
                                                   Value allocation) {
  return success();
}

LogicalResult comprehensiveBufferizeCopyFn(OpBuilder& builder, Location loc,
                                           Value from, Value to) {
  createLinalgCopyOp(builder, loc, from, to);
  return success();
}

OneShotBufferizationOptions getBufferizationOptions() {
  OneShotBufferizationOptions options;
  options.allocationFn = comprehenciveBufferizeAllocationFn;
  options.deallocationFn = comprehensiveBufferizeDeallocationFn;
  options.memCpyFn = comprehensiveBufferizeCopyFn;
  options.bufferAlignment = 64;
  options.createDeallocs = false;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocs = true;
  options.functionBoundaryTypeConversion =
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap;

  // bufferization.to_memref is used to bufferize constant_wrapper ops. DISC has
  // it's own logic to handle constants. We'd like to leave the these constant
  // ops as is and insert bufferization.to_memref to convert the tensor to
  // memref.
  options.opFilter.denyOperation<disc_linalg_ext::ConstantWrapperOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, unsigned memorySpace,
                                      const BufferizationOptions& options) {
    auto tensorType = value.getType().cast<TensorType>();
    return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                memorySpace);
  };

  return options;
}

/// Pattern to convert tensor.empty to bufferizaiton::AllocTensorOp.
struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};

LogicalResult bufferizeTensorEmptyOps(
    transform::TransformState& state, ModuleOp moduleOp,
    const OneShotBufferizationOptions& options) {
  /// 1, first convert tensor.emtpy -> TensorAlloc
  RewritePatternSet patterns(moduleOp->getContext());
  patterns.add<EmptyTensorLoweringPattern>(patterns.getContext());
  TrackingListener listener(state);
  GreedyRewriteConfig config;
  LogicalResult result = applyPatternsAndFoldGreedily(
      state.getTopLevel(), std::move(patterns), config, &listener);
  LogicalResult listenerResult = listener.checkErrorState();
  if (failed(result) || failed(listenerResult)) {
    return moduleOp->emitError() << "failed to bufferize tensor.empty\n";
  }
  /// 2, some TensorAlloc can be folded. Try to do some optimizations in such
  /// case.
  IRRewriter rewriter(moduleOp->getContext());
  OneShotAnalysisState oneShotState(moduleOp, options);
  if (failed(bufferization::analyzeOp(moduleOp, oneShotState)) ||
      failed(bufferization::insertSliceAnchoredAllocTensorEliminationStep(
          rewriter, moduleOp, oneShotState))) {
    return moduleOp->emitError()
           << "failed to analyze the module for bufferization\n";
  }

  return success();
}

}  // namespace

DiagnosedSilenceableFailure DISCBufferizeOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> payload = state.getPayloadOps(getTarget());
  if (payload.size() != 1 || !isa<ModuleOp>(payload.front())) {
    return mlir::emitDefiniteFailure(
        state.getTopLevel(), "requires exactly a single ModuleOp target op.");
  }

  auto moduleOp = cast<ModuleOp>(payload.front());
  auto options = getBufferizationOptions();

  // Bufferize tensor.empty
  if (failed(bufferizeTensorEmptyOps(state, moduleOp, options))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to bufferize tensor.empty.");
  }

  if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options)))
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "bufferization failed.");

  PassManager pm(getContext());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createPromoteBuffersToStackPass([](Value alloc) {
        return betterToUseAlloca(alloc.getType().cast<ShapedType>());
      }));
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());

  if (failed(pm.run(moduleOp)))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.set(getResult().cast<OpResult>(), {payload});

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//

namespace {

bool allEquals(ArrayAttr attrs, int val) {
  return llvm::all_of(attrs, [&](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt() == val;
  });
}

bool allConstIndexValues(ValueRange vals, int val) {
  return llvm::all_of(vals, [&](Value indexVal) {
    auto constOp = indexVal.getDefiningOp<arith::ConstantIndexOp>();
    return constOp && constOp.getValue().cast<IntegerAttr>().getInt() == val;
  });
}

bool offsetAllZeros(tensor::ExtractSliceOp slice) {
  return allEquals(slice.getStaticOffsets(), 0);
}

bool offsetAllZeros(tensor::InsertSliceOp slice) {
  return allEquals(slice.getStaticOffsets(), 0);
}

bool strideAllOnes(tensor::ExtractSliceOp slice) {
  auto strides = slice.getStaticStrides();
  return allEquals(strides, 1);
}

bool strideAllOnes(tensor::InsertSliceOp slice) {
  return allEquals(slice.getStaticStrides(), 1);
}

bool hasSameShape(ArrayRef<OpFoldResult> lhs, ArrayRef<OpFoldResult> rhs) {
  for (const auto& z : llvm::zip(lhs, rhs)) {
    auto lhsAttr = std::get<0>(z).dyn_cast<Attribute>();
    auto rhsAttr = std::get<1>(z).dyn_cast<Attribute>();
    if ((lhsAttr == nullptr) != (rhsAttr == nullptr)) return false;

    if (lhsAttr) {
      if (lhsAttr.cast<IntegerAttr>().getInt() !=
          rhsAttr.cast<IntegerAttr>().getInt())
        return false;
    } else {
      if (std::get<0>(z).dyn_cast<Value>() != std::get<0>(z).dyn_cast<Value>())
        return false;
    }
  }
  return true;
}

bool hasSameShape(tensor::ExtractSliceOp lhs, tensor::ExtractSliceOp rhs) {
  return hasSameShape(lhs.getMixedSizes(), rhs.getMixedSizes());
}

Value mapResultToInitOperandOfDestinationStyleOp(Value result) {
  auto dstOp = result.getDefiningOp<DestinationStyleOpInterface>();
  if (!dstOp) return nullptr;
  int resultNumber = result.cast<OpResult>().getResultNumber();
  return dstOp.getDpsInitOperand(resultNumber)->get();
}

tensor::ExtractSliceOp getSliceProducer(Value result) {
  if (!result) return nullptr;
  if (auto slice = result.getDefiningOp<tensor::ExtractSliceOp>()) return slice;
  return getSliceProducer(mapResultToInitOperandOfDestinationStyleOp(result));
}

/// Pattern to fold tensor.extract_slice of tensor.extract_slice.
/// convert:
///   %0 = tensor.extract_slice %arg0[%0, %t1][%s0, %s1][%c1, %c1]
///   %1 = tensor.extract_slice %0[%t2, %0][%s2, %s3][%c1, %c1]
/// to:
///   %1 = tensor.extract_slice %0[%t2, %t1][%s2, %s3][%c1, %c1]
struct FoldExtractSliceOfExtractSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto producerOp = op.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!strideAllOnes(op) || !producerOp || !strideAllOnes(producerOp))
      return failure();

    SmallVector<OpFoldResult> newOffsets;
    for (const auto& z :
         llvm::zip(op.getMixedOffsets(), producerOp.getMixedOffsets())) {
      auto lhsAttr = std::get<0>(z).dyn_cast<Attribute>();
      auto rhsAttr = std::get<1>(z).dyn_cast<Attribute>();
      if (!lhsAttr && !rhsAttr) return failure();
      IntegerAttr intLhsAttr, intRhsAttr;
      if (lhsAttr) intLhsAttr = lhsAttr.cast<IntegerAttr>();
      if (rhsAttr) intRhsAttr = rhsAttr.cast<IntegerAttr>();
      if (intLhsAttr && intRhsAttr) {
        newOffsets.push_back(IntegerAttr::get(
            intLhsAttr.getType(), intLhsAttr.getInt() + intRhsAttr.getInt()));
      } else if (intLhsAttr && intLhsAttr.getInt() == 0) {
        newOffsets.push_back(std::get<1>(z));
      } else if (intRhsAttr && intRhsAttr.getInt() == 0) {
        newOffsets.push_back(std::get<0>(z));
      } else {
        return failure();
      }
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, producerOp.getSource(), newOffsets, op.getMixedSizes(),
        op.getMixedStrides());
    return success();
  }
};

/// Fold a tensor.extract_slice op if we can know that it's a full select slice.
///
/// convert:
///   %0 = tensor.extract_slice %arg0[%o0, %o1][%s0, %s1][1, 1]
///   %1 = destination_style_op(%0)
///   %2 = tensor.extract_slice %1[0, 0][%s0, %s1][1, 1] // full select
///   use(%2)
/// to:
///   %0 = tensor.extract_slice %arg0[%o0, %o1][%s0, %s1][1, 1]
///   %1 = destination_style_op(%0)
///   use(%1)
struct FoldFullSelectedExtractSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    if (!offsetAllZeros(op) || !strideAllOnes(op)) return failure();

    auto sliceOp = getSliceProducer(op.getSource());
    if (!sliceOp || !hasSameShape(op, sliceOp)) return failure();
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

/// Fold a self assigned tensor.insert_slice op.
///
/// convert:
///   %0 = tensor.extract_slice %arg0[%o0, %o1][%s0, %s1][1, 1]
///   %1 = destination_style_op(%0)
///   %2 = vector.transfer_write %vec, %1[%c0, %c0]
///   %3 = tensor.insert_slice %2 into %1[0, 0] [%s0, %s1] [1, 1]
///   use(%3)
/// to:
///   %0 = tensor.extract_slice %arg0[%o0, %o1][%s0, %s1][1, 1]
///   %1 = destination_style_op(%0)
///   %2 = vector.transfer_write %vec, %1[%c0, %c0]
///   use(%2)
struct FoldSelfInsertSlicePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter& rewriter) const override {
    if (!offsetAllZeros(op) || !strideAllOnes(op)) return failure();

    auto sourceSliceOp = getSliceProducer(op.getSource());
    auto destSliceOp = getSliceProducer(op.getDest());
    if (!sourceSliceOp || !destSliceOp) return failure();
    // 1, dest & source have the same shape
    if (!hasSameShape(sourceSliceOp, destSliceOp)) return failure();
    // 2, the inserted tile from source and the dest have the same shape.
    if (!hasSameShape(sourceSliceOp.getMixedSizes(), op.getMixedSizes()))
      return failure();

    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

/// convert:
///   %0 = tensor.extract_slice %arg0[0, 0] [%s0, %s1] [1, 1] : tensor<?x?xf32>
///   to tensor<?x?xf32> %1 = linalg.fill ins(%cst : f32) outs(%0 :
///   tensor<?x?xf32>) -> tensor<?x?xf32>
///   %2 = vector.transfer_read %1[0, 0],
///            %cst : tensor<?x?xf32>, vector<6x16xf32>
/// to:
///   %2 = vector.splat %cst : vector<6x16xf32>
struct TransferReadOfFillOpPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter& rewriter) const override {
    // offsets are zeros.
    if (!allConstIndexValues(op.getIndices(), 0)) return failure();

    auto fillOp = op.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp || fillOp->getOperand(0) != op.getPadding()) return failure();
    rewriter.replaceOpWithNewOp<vector::SplatOp>(op, op.getType(),
                                                 op.getPadding());
    return success();
  }
};

/// convert:
///   %0 = tensor.extract_slice %arg0[0, 0] [%s0, %s1] [1, 1]
///       : tensor<?x?xf32> to tensor<?x?xf32>
///   %1 = linalg.fill ins(%cst : f32) outs(%0
///       : tensor<?x?xf32>) -> tensor<?x?xf32>
///   %2 = vector.transfer_write %result, %1[0, 0]
///                     : vector<6x16xf32>, tensor<?x?xf32>
/// to:
///   %2 = vector.transfer_write %result, %0[0, 0] : vector<6x16xf32>
/// iff:
///   %1 has only one user
///   %s0 < dim 0 of %result and %s1 < dim 1 of %result
struct TransferWriteOfFillOpPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter& rewriter) const override {
    // offsets are zeros.
    if (!allConstIndexValues(op.getIndices(), 0)) return failure();

    auto fillOp = op.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp || !fillOp.result().hasOneUse()) return failure();

    auto sliceOp = fillOp.output().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp || !strideAllOnes(sliceOp)) return failure();

    auto type = op.getVector().getType().cast<VectorType>();
    for (const auto& z : llvm::zip(sliceOp.getMixedSizes(), type.getShape())) {
      if (std::get<1>(z) == ShapedType::kDynamicSize) return failure();
      if (auto attr = std::get<0>(z).dyn_cast<Attribute>()) {
        if (attr.cast<IntegerAttr>().getInt() > std::get<1>(z))
          return failure();
      } else {
        auto bound = linalg::getConstantUpperBoundForIndex(
            std::get<0>(z).dyn_cast<Value>());
        if (failed(bound) || *bound > std::get<1>(z)) return failure();
      }
    }

    auto cloned = rewriter.clone(*op.getOperation());
    cloned->setOperand(1, fillOp.output());
    rewriter.replaceOp(op, cloned->getResults());
    return success();
  }
};

/// convert:
///   %0 = disc_linalg_ext.constant_wrapper ...
///   %1 = disc_linalg_ext.multi_level_pack %0 ...
///   use(%1)
/// to:
///   %0 = disc_linalg_ext.constant_wrapper ... // folded
///   use(%0)
struct FoldMultiLevelPackOfConstantWrapperPattern
    : public OpRewritePattern<MultiLevelPackOp> {
  using OpRewritePattern<MultiLevelPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiLevelPackOp op,
                                PatternRewriter& rewriter) const override {
    auto constOp = op.getInput().getDefiningOp<ConstantWrapperOp>();
    if (!constOp) return failure();

    Attribute paddingAttr;
    if (op.getPaddingValue() &&
        !matchPattern(op.getPaddingValue(), m_Constant(&paddingAttr)))
      return failure();

    SmallVector<Attribute> attrs{constOp.getValue(), nullptr, paddingAttr};
    SmallVector<OpFoldResult> results;
    if (failed(op.fold(attrs, results))) return failure();

    rewriter.replaceOpWithNewOp<ConstantWrapperOp>(op, op.getOutputType(),
                                                   results[0].get<Attribute>());
    return success();
  }
};

static void addAllRegisteredCanonicalizationPatterns(
    RewritePatternSet& patterns) {
  MLIRContext* ctx = patterns.getContext();
  for (Dialect* dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  patterns.insert<FoldExtractSliceOfExtractSlicePattern>(ctx);
  patterns.insert<FoldFullSelectedExtractSlicePattern>(ctx);
  patterns.insert<FoldSelfInsertSlicePattern>(ctx);
  patterns.insert<TransferReadOfFillOpPattern>(ctx);
  patterns.insert<TransferWriteOfFillOpPattern>(ctx);
  patterns.insert<FoldMultiLevelPackOfConstantWrapperPattern>(ctx);
}

}  // namespace

void ApplyPatternsOp::build(OpBuilder& builder, OperationState& result,
                            Value target, bool canonicalization) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  if (canonicalization) {
    result.addAttribute(
        ApplyPatternsOp::getCanonicalizationAttrName(result.name),
        builder.getUnitAttr());
  }
  result.addTypes({pdl::OperationType::get(ctx)});
}

DiagnosedSilenceableFailure ApplyPatternsOp::applyToOne(
    Operation* target, SmallVectorImpl<Operation*>& results,
    transform::TransformState& state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(
        target,
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
  }
  MLIRContext* ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);

  TrackingListener listener(state);
  GreedyRewriteConfig config;
  LogicalResult result = applyPatternsAndFoldGreedily(
      target, std::move(patterns), config, &listener);
  LogicalResult listenerResult = listener.checkErrorState();
  if (failed(result)) {
    return mlir::emitDefiniteFailure(target,
                                     "greedy pattern application failed");
  }
  if (failed(listenerResult))
    return mlir::emitDefiniteFailure(target, "listener tracking failed");

  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// FoldProducerExtractSliceOp
//===---------------------------------------------------------------------===//

namespace {

LogicalResult tryMergeExtractSliceOp(OpBuilder& b,
                                     tensor::ExtractSliceOp& sliceOp,
                                     tensor::ExtractSliceOp producerOp) {
  if (!strideAllOnes(sliceOp) || !strideAllOnes(producerOp)) return failure();
  Location loc = sliceOp->getLoc();
  SmallVector<OpFoldResult> newOffsets;
  for (const auto& z :
       llvm::zip(sliceOp.getMixedOffsets(), producerOp.getMixedOffsets())) {
    Value lhsValue = std::get<0>(z).dyn_cast<Value>();
    Value rhsValue = std::get<1>(z).dyn_cast<Value>();
    if (!lhsValue) {
      lhsValue = b.create<arith::ConstantIndexOp>(
          loc,
          std::get<0>(z).dyn_cast<Attribute>().cast<IntegerAttr>().getInt());
    }
    if (!rhsValue) {
      rhsValue = b.create<arith::ConstantIndexOp>(
          loc,
          std::get<1>(z).dyn_cast<Attribute>().cast<IntegerAttr>().getInt());
    }
    newOffsets.push_back(
        b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult());
  }
  sliceOp = b.create<tensor::ExtractSliceOp>(
      loc, producerOp.getSource(), newOffsets, sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  return success();
}

}  // namespace

void FoldProducerExtractSliceOp::build(OpBuilder& builder,
                                       OperationState& result, Value target,
                                       int64_t maxRepeatNum) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      FoldProducerExtractSliceOp::getMaxRepeatNumAttrName(result.name),
      builder.getIntegerAttr(builder.getIntegerType(64), maxRepeatNum));
  result.addTypes({pdl::OperationType::get(ctx)});
}

DiagnosedSilenceableFailure FoldProducerExtractSliceOp::applyToOne(
    Operation* target, SmallVectorImpl<Operation*>& results,
    transform::TransformState& state) {
  auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(target);
  if (!sliceOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to tensor::ExtractSliceOp");
  }
  MLIRContext* ctx = target->getContext();

  OpBuilder b(target);
  for (int64_t i = 0; i < getMaxRepeatNum(); ++i) {
    auto producerOp =
        sliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!producerOp || failed(tryMergeExtractSliceOp(b, sliceOp, producerOp)))
      break;
  }

  if (sliceOp.getOperation() != target) {
    target->getResult(0).replaceAllUsesWith(sliceOp.getResult());
  }

  results.assign({sliceOp.getOperation()});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// CacheReadOp
//===---------------------------------------------------------------------===//

namespace {

// Suppose:
//   - `di` is one dimension of the source tensor.
//   - `dj` is one of the tiled dimension of `di`.
// A dimension `dj` of the packed tensor is the inner most dim if
//   - the tile level of `di` is 0, or
//   - dj is corresponding to the last tile size of `di`.
struct InnerMostDimsInfo {
  // the dimension indices for the inner most dimensions.
  SmallVector<int64_t> dims;
  // the dimension sizes for the inner most dimensions.
  SmallVector<int64_t> dimSizes;
  // dimSizes after being transposed
  SmallVector<int64_t> transposedDimSizes;
  // permutation for the inner most dimensions.
  SmallVector<int64_t> permutation;
  // reverse permutation for the inner most dimensions.
  SmallVector<int64_t> reversePermutation;

  // Returns true if the dimension `d` is the inner most.
  bool isInnerMostDim(int d) { return llvm::find(dims, d) != dims.end(); }
};

void parseInnerMostDimsInfo(InnerMostDimsInfo& info, ShapedType sourceTy,
                            ArrayRef<int64_t> tileLevelsVec,
                            ArrayRef<int64_t> tileSizesVec,
                            ArrayRef<int64_t> permutationVec) {
  int64_t totalNumDims = -1;
  for (const auto& en : llvm::enumerate(tileLevelsVec)) {
    totalNumDims += en.value() + 1;
    info.dims.push_back(totalNumDims);
    if (en.value() == 0) {
      info.dimSizes.push_back(sourceTy.getShape()[en.index()]);
    } else {
      info.dimSizes.push_back(tileSizesVec[totalNumDims - en.index() - 1]);
    }
  }

  for (int64_t d : permutationVec) {
    auto it = llvm::find(info.dims, d);
    if (it == info.dims.end()) continue;
    info.permutation.push_back(std::distance(info.dims.begin(), it));
  }

  info.reversePermutation = info.permutation;
  for (int i = 0; i < tileLevelsVec.size(); ++i) {
    info.reversePermutation[info.permutation[i]] = i;
  }

  info.transposedDimSizes =
      disc_linalg_ext::interchange<int64_t>(info.dimSizes, info.permutation);
}

linalg::GenericOp makeTransposeOp(OpBuilder& b, Location loc, Value inputTensor,
                                  Value outputTensor,
                                  ArrayRef<int64_t> transposeVector) {
  auto resultTensorType = outputTensor.getType().cast<RankedTensorType>();
  Type elementType = resultTensorType.getElementType();

  // Compute the transpose and the indentity indexing maps.
  SmallVector<AffineMap> indexingMaps = {
      inversePermutation(AffineMap::getPermutationMap(
          SmallVector<unsigned>(transposeVector.begin(), transposeVector.end()),
          b.getContext())),
      AffineMap::getMultiDimIdentityMap(transposeVector.size(),
                                        b.getContext())};
  SmallVector<llvm::StringRef> iteratorTypes(transposeVector.size(),
                                             getParallelIteratorTypeName());

  // Create a GenericOp to transpose `inputTensor` into `outputTensor`.
  auto transposeOp = b.create<linalg::GenericOp>(
      loc, resultTensorType, inputTensor, outputTensor,
      b.getAffineMapArrayAttr(indexingMaps), b.getStrArrayAttr(iteratorTypes),
      /*doc=*/nullptr,
      /*library_call=*/nullptr);
  Region& body = transposeOp.getRegion();
  body.push_back(new Block());
  body.front().addArguments({elementType, elementType}, {loc, loc});

  // Create the body of the transpose operation.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&body.front());
  b.create<linalg::YieldOp>(loc,
                            transposeOp.getRegion().front().getArgument(0));
  return transposeOp;
}

LogicalResult readFromPackedValue(
    OpBuilder& b, Location loc, Value packedValue, ShapedType targetTy,
    ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> tileLevelsVec,
    ArrayRef<int64_t> tileSizesVec, ArrayRef<int64_t> permutationVec,
    InnerMostDimsInfo& info, Operation*& resultOp) {
  auto packedTy = packedValue.getType().cast<ShapedType>();
  int packedRank = packedTy.getRank();

  auto const2IndexAttr = [&](int64_t val) {
    return IntegerAttr::get(b.getIndexType(), val);
  };

  SmallVector<OpFoldResult> newOffsets(packedRank);
  SmallVector<OpFoldResult> newSizes(packedRank);
  SmallVector<OpFoldResult> newStrides(packedRank,
                                       OpFoldResult{const2IndexAttr(1)});

  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  AffineExpr floorDivExpr = s0.floorDiv(s1);
  AffineExpr modExpr = s0 % s1;

  int tileSizeIdx = 0;
  int resultDimIdx = 0;
  for (int dimIdx = 0; dimIdx < offsets.size(); ++dimIdx) {
    int64_t level = tileLevelsVec[dimIdx];
    OpFoldResult offset = offsets[dimIdx];
    for (int localResultDimIdx = 0; localResultDimIdx <= level;
         ++localResultDimIdx) {
      int d = resultDimIdx + localResultDimIdx;
      if (info.isInnerMostDim(d)) {
        newOffsets[d] = const2IndexAttr(0);
        newSizes[d] = const2IndexAttr(info.dimSizes[dimIdx]);
      } else {
        newSizes[d] = const2IndexAttr(1);
        OpFoldResult tileSize =
            const2IndexAttr(tileSizesVec[tileSizeIdx + localResultDimIdx]);
        newOffsets[d] = makeComposedFoldedAffineApply(b, loc, floorDivExpr,
                                                      {offset, tileSize});
        offset =
            makeComposedFoldedAffineApply(b, loc, modExpr, {offset, tileSize});
      }
    }
    tileSizeIdx += level;
    resultDimIdx += 1 + level;
  }

  if (!permutationVec.empty()) {
    newOffsets =
        disc_linalg_ext::interchange<OpFoldResult>(newOffsets, permutationVec);
    newSizes =
        disc_linalg_ext::interchange<OpFoldResult>(newSizes, permutationVec);
  }

  auto sliceTy =
      RankedTensorType::get(info.transposedDimSizes, packedTy.getElementType());
  auto sliceOp = b.create<tensor::ExtractSliceOp>(
      loc, sliceTy, packedValue, newOffsets, newSizes, newStrides);
  if (sliceTy != targetTy) {
    Value emptyResult = b.create<tensor::EmptyOp>(loc, targetTy, ValueRange{});
    resultOp =
        makeTransposeOp(b, loc, sliceOp, emptyResult, info.reversePermutation);
  } else {
    resultOp = sliceOp.getOperation();
  }
  return success();
}

}  // namespace

void CacheReadOp::build(OpBuilder& builder, OperationState& result,
                        Value target, Value anchor,
                        ArrayRef<int64_t> tileLevels,
                        ArrayRef<int64_t> tileSizes, bool padded,
                        ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> permutationVec;
  int64_t expectedResultRank =
      disc_linalg_ext::MultiLevelPackOp::getExpectedResultRank(tileLevels);
  if (expectedResultRank > 0 && permutation.empty()) {
    permutationVec = llvm::to_vector<>(
        llvm::seq(static_cast<int64_t>(0), expectedResultRank));
    permutation = permutationVec;
  }
  build(builder, result, pdl::OperationType::get(builder.getContext()), target,
        anchor, builder.getI64ArrayAttr(tileLevels),
        builder.getI64ArrayAttr(tileSizes),
        builder.getI64ArrayAttr(permutation));
  if (padded) {
    result.addAttribute(CacheReadOp::getPaddedAttrName(result.name),
                        builder.getUnitAttr());
  }
}

void CacheReadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::onlyReadsHandle({this->getOperation()->getOperand(1)}, effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure CacheReadOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one target op but got ")
           << targetOps.size();

  Operation* targetOp = targetOps[0];
  tensor::ExtractSliceOp sliceOp;
  Optional<Value> paddingValue;
  if (getPadded()) {
    auto padOp = dyn_cast<tensor::PadOp>(targetOp);
    if (!padOp)
      return mlir::emitDefiniteFailure(targetOp,
                                       "expect target to be pad op when "
                                       "`padded` attr is set, but got ")
             << *targetOp;
    sliceOp = padOp->getOperand(0).getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      return mlir::emitDefiniteFailure(
          targetOp,
          "expect the source of pad is a slice op when `padded` attr is set");
    if (padOp.getBody()->getOperations().size() != 1)
      return mlir::emitDefiniteFailure(
          targetOp, "expect the padding region of pad op only has one op\n");
    paddingValue =
        cast<tensor::YieldOp>(&padOp.getBody()->getOperations().front())
            ->getOperand(0);
  } else {
    sliceOp = dyn_cast<tensor::ExtractSliceOp>(targetOp);
    if (!sliceOp)
      return mlir::emitDefiniteFailure(
                 targetOp,
                 "expect target to be extract_slice op when `padded` attr is "
                 "not set, but got ")
             << *targetOp;
  }

  if (!strideAllOnes(sliceOp))
    return mlir::emitDefiniteFailure(
        sliceOp.getOperation(),
        "expect the strides of target slice op are all ones\n");

  // verify the target op have static shape
  auto targetTy = targetOp->getResult(0).getType().cast<ShapedType>();
  if (!targetTy.hasStaticShape() || targetTy.getRank() == 0)
    return mlir::emitDefiniteFailure(
        targetOp, "expect the targetOp has static shape with rank > 0\n");

  ArrayRef<Operation*> anchorOps = state.getPayloadOps(getAnchor());
  if (anchorOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one anchor op but got ")
           << anchorOps.size();

  Operation* anchorOp = anchorOps[0];
  OpBuilder b(anchorOp);
  Location loc = anchorOp->getLoc();
  Value source = sliceOp.getSource();
  auto sourceTy = source.getType().cast<ShapedType>();
  auto tileLevelsVec =
      MultiLevelPackOp::convertI64ArrayAttrToVec(getTileLevels());
  auto tileSizesVec =
      MultiLevelPackOp::convertI64ArrayAttrToVec(getTileSizes());
  auto permutationVec =
      MultiLevelPackOp::convertI64ArrayAttrToVec(getPermutation());
  auto packedType = MultiLevelPackOp::getPackedType(
      sourceTy, tileLevelsVec, tileSizesVec, permutationVec);
  if (!packedType)
    return mlir::emitDefiniteFailure(targetOp,
                                     "failed to infer the packed type\n");

  // Verify that: the inner most tile size (or the dim size if the dimension is
  // not tiled at all) for each dimension equals to the dimension size of the
  // (padded) slice correspondingly.
  InnerMostDimsInfo innerMostDimsInfo;
  parseInnerMostDimsInfo(innerMostDimsInfo, sourceTy, tileLevelsVec,
                         tileSizesVec, permutationVec);
  for (const auto& z :
       llvm::zip(targetTy.getShape(), innerMostDimsInfo.dimSizes)) {
    if (std::get<0>(z) != std::get<1>(z))
      return mlir::emitDefiniteFailure(this->getOperation(),
                                       "expect the inner most tile size match "
                                       "the shape of the result of slice: ")
             << std::get<0>(z) << " vs " << std::get<1>(z) << "\n";
  }

  SmallVector<OpFoldResult> sourceDims =
      disc_linalg_ext::getDims(b, loc, source);
  auto resultDims = MultiLevelPackOp::getResultShape(
      b, loc, sourceDims, tileLevelsVec, tileSizesVec, permutationVec);
  SmallVector<Value> resultDynDims;
  for (auto r : resultDims)
    if (auto v = r.dyn_cast<Value>()) resultDynDims.push_back(v);
  auto empty = b.create<tensor::EmptyOp>(loc, packedType, resultDynDims);
  auto packedOp =
      b.create<MultiLevelPackOp>(loc, source, empty, tileLevelsVec,
                                 tileSizesVec, permutationVec, paddingValue);

  b.setInsertionPoint(targetOp);
  Operation* resultOp;
  if (failed(readFromPackedValue(b, loc, packedOp->getResult(0), targetTy,
                                 sliceOp.getMixedOffsets(), tileLevelsVec,
                                 tileSizesVec, permutationVec,
                                 innerMostDimsInfo, resultOp)))
    return mlir::emitDefiniteFailure(
        this->getOperation(),
        "failed to create new extract_slice op for the packed value\n");
  targetOp->getResult(0).replaceAllUsesWith(resultOp->getResult(0));
  results.set(getResult().cast<OpResult>(), {resultOp});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// LowerMultiLevelPackToLoopOp
//===---------------------------------------------------------------------===//

namespace {

SmallVector<Value> getDimValues(OpBuilder& b, Location loc, Value v) {
  auto ty = v.getType().cast<RankedTensorType>();
  SmallVector<Value> vs;
  for (int64_t i = 0; i < ty.getRank(); ++i)
    vs.push_back(b.create<tensor::DimOp>(loc, v, i));
  return vs;
}

Value buildMin(OpBuilder& b, Location loc, Value lhs, Value rhs) {
  SmallVector<OpFoldResult> vals = {lhs, rhs};
  auto result = makeComposedFoldedAffineMin(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
  return getValueOrCreateConstantIndexOp(b, loc, result);
}

}  // namespace

void LowerMultiLevelPackToLoopOp::build(OpBuilder& builder,
                                        OperationState& result, Value target) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  result.addTypes({pdl::OperationType::get(ctx)});
}

DiagnosedSilenceableFailure LowerMultiLevelPackToLoopOp::applyToOne(
    Operation* target, SmallVectorImpl<Operation*>& results,
    transform::TransformState& state) {
  auto multiLevelPackOp = dyn_cast<MultiLevelPackOp>(target);
  if (!multiLevelPackOp) {
    return mlir::emitDefiniteFailure(
        target, "applies only to disc_linalg_ext::MultiLevelPackOp");
  }

  OpBuilder b(target);
  Location loc = target->getLoc();
  MLIRContext* ctx = target->getContext();
  auto tileLevelsVec = multiLevelPackOp.getTileLevelsVec();
  auto tileSizesVec = multiLevelPackOp.getTileSizesVec();
  auto permutationVec = multiLevelPackOp.getPermutationVec();

  Value src = multiLevelPackOp.getInput();
  auto srcTy = src.getType().cast<RankedTensorType>();
  int64_t srcRank = srcTy.getRank();
  Value dst = multiLevelPackOp.getOutput();
  auto dstTy = dst.getType().cast<RankedTensorType>();
  int64_t dstRank = dstTy.getRank();

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> srcOffsets(srcRank, zero);
  SmallVector<Value> srcSizes = getDimValues(b, loc, src);
  SmallVector<Value> srcStrides(srcRank, one);
  SmallVector<Value> dstOffsets(dstRank, zero);
  SmallVector<Value> dstSizes = getDimValues(b, loc, dst);
  SmallVector<Value> dstStrides(dstRank, one);

  InnerMostDimsInfo innerMostDimsInfo;
  parseInnerMostDimsInfo(innerMostDimsInfo, srcTy, tileLevelsVec, tileSizesVec,
                         permutationVec);
  auto logicalDim2SrcDim =
      multiLevelPackOp.getOutputLogicalDimToInputDimMapping(tileLevelsVec,
                                                            tileSizesVec);
  auto logicalDim2TileSize =
      multiLevelPackOp.getOutputLogicalDimToTileSizeMapping(tileLevelsVec,
                                                            tileSizesVec);

  Value loopInitValue = dst;
  SmallVector<scf::ForOp> forOps;
  SmallVector<Value> srcDimUppers = srcSizes;
  auto staticSrcDimUppers = llvm::to_vector<>(srcTy.getShape());
  for (int dstIdx = 0; dstIdx < dstRank; ++dstIdx) {
    int logicalIdx = permutationVec[dstIdx];
    if (innerMostDimsInfo.isInnerMostDim(logicalIdx)) continue;
    int64_t staticStep = logicalDim2TileSize[logicalIdx];
    Value step = b.create<arith::ConstantIndexOp>(loc, staticStep);
    int srcIdx = logicalDim2SrcDim[logicalIdx];
    Value upper = srcDimUppers[srcIdx];
    auto forOp =
        b.create<scf::ForOp>(loc, zero, upper, step, ValueRange{loopInitValue});
    b.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
    Value iv = forOp.getInductionVar();
    srcOffsets[srcIdx] = b.create<arith::AddIOp>(loc, srcOffsets[srcIdx], iv);
    if (staticStep == 1 ||
        staticSrcDimUppers[srcIdx] != ShapedType::kDynamicSize &&
            staticSrcDimUppers[srcIdx] % staticStep == 0) {
      srcDimUppers[srcIdx] = step;
      staticSrcDimUppers[srcIdx] = staticStep;
    } else {
      Value remaining = b.create<arith::SubIOp>(loc, upper, iv);
      srcDimUppers[srcIdx] = buildMin(b, loc, remaining, step);
      staticSrcDimUppers[srcIdx] = ShapedType::kDynamicSize;
    }
    dstOffsets[dstIdx] = b.create<arith::DivSIOp>(loc, iv, step);
    dstSizes[dstIdx] = one;
    forOps.push_back(forOp);
    loopInitValue = *forOp.getRegionIterArgs().begin();
  }

  Value srcSlice = b.create<tensor::ExtractSliceOp>(loc, src, srcOffsets,
                                                    srcDimUppers, srcStrides);
  if (Value paddingValue = multiLevelPackOp.getPaddingValue()) {
    SmallVector<OpFoldResult> lowIndices(srcRank,
                                         IntegerAttr::get(b.getIndexType(), 0));
    SmallVector<OpFoldResult> highIndices;
    for (const auto& z : llvm::zip(srcDimUppers, innerMostDimsInfo.dimSizes)) {
      Value paddedSize = b.create<arith::ConstantIndexOp>(loc, std::get<1>(z));
      highIndices.push_back(
          b.create<arith::SubIOp>(loc, paddedSize, std::get<0>(z)).getResult());
    }
    srcSlice = b.create<tensor::PadOp>(
        loc,
        RankedTensorType::get(innerMostDimsInfo.dimSizes,
                              srcTy.getElementType()),
        srcSlice, lowIndices, highIndices, paddingValue);
  }

  // transpose srcSlice when needed.
  auto sortedPermutation = innerMostDimsInfo.permutation;
  llvm::sort(sortedPermutation);
  if (sortedPermutation != innerMostDimsInfo.permutation) {
    auto transposeDstTy = RankedTensorType::get(
        innerMostDimsInfo.transposedDimSizes, dstTy.getElementType());
    Value transposeDst =
        b.create<tensor::ExtractSliceOp>(loc, transposeDstTy, loopInitValue,
                                         dstOffsets, dstSizes, dstStrides)
            ->getResult(0);
    srcSlice = makeTransposeOp(b, loc, srcSlice, transposeDst,
                               innerMostDimsInfo.permutation)
                   ->getResult(0);
  }

  Value updateDst =
      b.create<tensor::InsertSliceOp>(loc, srcSlice, loopInitValue, dstOffsets,
                                      dstSizes, dstStrides)
          ->getResult(0);
  b.create<scf::YieldOp>(loc, updateDst);
  for (int i = static_cast<int>(forOps.size()) - 2; i >= 0; --i) {
    b.setInsertionPointAfter(forOps[i + 1]);
    b.create<scf::YieldOp>(loc, forOps[i + 1]->getResult(0));
  }
  assert(forOps.size() > 0);
  target->getResult(0).replaceAllUsesWith(forOps[0]->getResult(0));

  results.assign({forOps[0]});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// InlineReductionInitializer
//===---------------------------------------------------------------------===//

namespace {

/// Given two MemRefTypes `aT` and `bT`, return a MemRefType to which both can
/// be cast. If the MemRefTypes don't have the same rank or are not strided,
/// return null; otherwise:
///   1. if `aT` and `bT` are cast-compatible, return `aT`.
///   2. else return a new MemRefType obtained by iterating over the shape and
///   strides and:
///     a. keeping the ones that are static and equal across `aT` and `bT`.
///     b. using a dynamic shape and/or stride for the dimensions that don't
///        agree.
MemRefType getCastCompatibleMemRefType(MemRefType aT, MemRefType bT) {
  if (memref::CastOp::areCastCompatible(aT, bT)) return aT;
  if (aT.getRank() != bT.getRank()) return MemRefType();
  int64_t aOffset, bOffset;
  SmallVector<int64_t, 4> aStrides, bStrides;
  if (failed(getStridesAndOffset(aT, aStrides, aOffset)) ||
      failed(getStridesAndOffset(bT, bStrides, bOffset)) ||
      aStrides.size() != bStrides.size())
    return MemRefType();

  ArrayRef<int64_t> aShape = aT.getShape(), bShape = bT.getShape();
  int64_t resOffset;
  SmallVector<int64_t, 4> resShape(aT.getRank(), 0),
      resStrides(bT.getRank(), 0);
  for (int64_t idx = 0, e = aT.getRank(); idx < e; ++idx) {
    resShape[idx] =
        (aShape[idx] == bShape[idx]) ? aShape[idx] : ShapedType::kDynamicSize;
    resStrides[idx] = (aStrides[idx] == bStrides[idx])
                          ? aStrides[idx]
                          : ShapedType::kDynamicStrideOrOffset;
  }
  resOffset =
      (aOffset == bOffset) ? aOffset : ShapedType::kDynamicStrideOrOffset;
  return MemRefType::get(
      resShape, aT.getElementType(),
      StridedLayoutAttr::get(aT.getContext(), resOffset, resStrides));
}

}  // namespace

void InlineReductionInitializerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::onlyReadsHandle({this->getOperation()->getOperand(1)}, effects);
  transform::consumesHandle({this->getOperation()->getOperand(2)}, effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure InlineReductionInitializerOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> initOps = state.getPayloadOps(getInitializer());
  if (initOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one initializer but got ")
           << initOps.size();
  auto fillOp = dyn_cast<linalg::FillOp>(initOps[0]);
  if (!fillOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the initializer be a fill op");

  ArrayRef<Operation*> loopOps = state.getPayloadOps(getLoop());
  if (loopOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one loop but got ")
           << loopOps.size();
  auto forOp = dyn_cast<scf::ForOp>(loopOps[0]);
  if (!forOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the loop to a for op");

  ArrayRef<Operation*> readerOps = state.getPayloadOps(getReader());
  if (readerOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one reader but got ")
           << readerOps.size();
  auto readerOp = dyn_cast<vector::TransferReadOp>(readerOps[0]);
  if (!readerOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the reader to a transfer_read op");

  OpBuilder b(fillOp.getOperation());
  auto vectorTy = readerOp->getResultTypes()[0].cast<VectorType>();
  auto memrefTy =
      MemRefType::get(vectorTy.getShape(), vectorTy.getElementType());
  Value newInitBuffer = b.create<memref::AllocaOp>(
      fillOp.getLoc(), memrefTy, ValueRange{}, b.getI64IntegerAttr(64));
  b.create<linalg::FillOp>(fillOp.getLoc(), ValueRange{readerOp.getPadding()},
                           ValueRange{newInitBuffer});

  b.setInsertionPoint(readerOp.getOperation());
  Value firstIter =
      b.create<arith::CmpIOp>(readerOp.getLoc(), arith::CmpIPredicate::eq,
                              forOp.getLowerBound(), forOp.getInductionVar());
  auto castTy = getCastCompatibleMemRefType(
      readerOp.getSource().getType().cast<MemRefType>(), memrefTy);
  if (!castTy) {
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "failed to find compatible memref type");
  }

  auto ifOp = b.create<scf::IfOp>(readerOp.getLoc(), TypeRange{castTy},
                                  firstIter, true);
  b.setInsertionPointToStart(ifOp.thenBlock());
  Value thenResult =
      b.create<memref::CastOp>(readerOp.getLoc(), castTy, newInitBuffer)
          ->getResult(0);
  b.create<scf::YieldOp>(readerOp.getLoc(), thenResult);
  b.setInsertionPointToStart(ifOp.elseBlock());
  Value elseResult =
      b.create<memref::CastOp>(readerOp.getLoc(), castTy, readerOp.getSource())
          ->getResult(0);
  b.create<scf::YieldOp>(readerOp.getLoc(), elseResult);
  readerOp->replaceUsesOfWith(readerOp.getSource(), ifOp->getResult(0));
  fillOp->erase();
  results.set(getResult().cast<OpResult>(), {readerOp});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// DecomposeVectorsOp
//===---------------------------------------------------------------------===//

namespace {

Optional<SmallVector<int64_t>> computeShapeRatio(ArrayRef<int64_t> superShape,
                                                 ArrayRef<int64_t> subShape) {
  if (superShape.size() < subShape.size()) {
    return Optional<SmallVector<int64_t>>();
  }

  // Starting from the end, compute the integer divisors.
  std::vector<int64_t> result;
  result.reserve(superShape.size());
  for (auto [superSize, subSize] :
       llvm::zip(llvm::reverse(superShape), llvm::reverse(subShape))) {
    assert(superSize > 0 && "superSize must be > 0");
    assert(subSize > 0 && "subSize must be > 0");

    // If integral division does not occur, return and let the caller decide.
    if (superSize % subSize != 0) return llvm::None;
    result.push_back(superSize / subSize);
  }

  // At this point we computed the ratio (in reverse) for the common
  // size. Fill with the remaining entries from the super-vector shape (still in
  // reverse).
  int commonSize = subShape.size();
  std::copy(superShape.rbegin() + commonSize, superShape.rend(),
            std::back_inserter(result));

  assert(result.size() == superShape.size() &&
         "super to sub shape ratio is not of the same size as the super rank");

  // Reverse again to get it back in the proper order and return.
  return SmallVector<int64_t>{result.rbegin(), result.rend()};
}

SmallVector<int64_t> computeStrides(ArrayRef<int64_t> shape,
                                    ArrayRef<int64_t> sizes) {
  int64_t rank = shape.size();
  // Compute the count for each dimension.
  SmallVector<int64_t> sliceDimCounts(rank);
  for (int64_t r = 0; r < rank; ++r)
    sliceDimCounts[r] = ceilDiv(shape[r], sizes[r]);
  // Use that to compute the slice stride for each dimension.
  SmallVector<int64_t> sliceStrides(rank);
  sliceStrides[rank - 1] = 1;
  for (int64_t r = rank - 2; r >= 0; --r)
    sliceStrides[r] = sliceStrides[r + 1] * sliceDimCounts[r + 1];
  return sliceStrides;
}

SmallVector<int64_t> computeElementOffsetsFromVectorSliceOffsets(
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> vectorOffsets) {
  SmallVector<int64_t> result;
  for (auto it : llvm::zip(vectorOffsets, sizes))
    result.push_back(std::get<0>(it) * std::get<1>(it));
  return result;
}

/// During unrolling from `originalShape` to `targetShape` return the offset for
/// the slice `index`.
static SmallVector<int64_t> getVectorOffset(ArrayRef<int64_t> originalShape,
                                            ArrayRef<int64_t> targetShape,
                                            int64_t index) {
  SmallVector<int64_t> dstSliceStrides =
      computeStrides(originalShape, targetShape);
  SmallVector<int64_t> vectorOffsets = delinearize(dstSliceStrides, index);
  SmallVector<int64_t> elementOffsets =
      computeElementOffsetsFromVectorSliceOffsets(targetShape, vectorOffsets);
  return elementOffsets;
}

struct DecomposeVectorInOutsOfForOp : public OpRewritePattern<scf::ForOp> {
  explicit DecomposeVectorInOutsOfForOp(
      MLIRContext* context, const vector::UnrollVectorOptions& options,
      PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<scf::ForOp>(context, benefit),
        options_(options) {}

  struct VectorDecomposeInfo {
    VectorType dstVecType;
    int64_t startIdx;
    int64_t numSubVectors;
    SmallVector<SmallVector<int64_t>> strides;
    SmallVector<SmallVector<int64_t>> offsets;
  };

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter& rewriter) const final {
    Location loc = forOp.getLoc();
    if (failed(options_.filterConstraint(forOp))) return failure();
    auto maybeTargetShape = options_.nativeShape(forOp);
    if (!maybeTargetShape) return failure();
    auto& targetShape = *maybeTargetShape;
    int numNewInitArgs = 0;
    VectorType targetVectorType;
    DenseMap<int, VectorDecomposeInfo> candidateValueMap;
    for (const auto& en : llvm::enumerate(forOp.getInitArgs())) {
      auto ty = en.value().getType().dyn_cast<VectorType>();
      Optional<SmallVector<int64_t>> maybeShapeRatio;
      if (!ty ||
          !(maybeShapeRatio = computeShapeRatio(ty.getShape(), targetShape)) ||
          llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
        ++numNewInitArgs;
        continue;
      }
      // TODO(wyzero): support multiple iter args with different vector type.
      if (targetVectorType && targetVectorType != ty) return failure();
      targetVectorType = ty;
      auto& item = candidateValueMap[en.index()];
      item.dstVecType = ty;
      item.numSubVectors = computeMaxLinearIndex(*maybeShapeRatio);
      item.startIdx = numNewInitArgs;
      SmallVector<int64_t> strides(targetShape.size(), 1);
      for (int i = 0; i < item.numSubVectors; ++i) {
        auto offsets = getVectorOffset(ty.getShape(), targetShape, i);
        item.strides.push_back(strides);
        item.offsets.push_back(offsets);
        ++numNewInitArgs;
      }
    }
    if (candidateValueMap.empty()) return failure();

    SmallVector<Value> newIterArgs;
    for (const auto& en : llvm::enumerate(forOp.getInitArgs())) {
      auto it = candidateValueMap.find(en.index());
      if (it == candidateValueMap.end()) {
        newIterArgs.push_back(en.value());
      } else {
        auto& item = it->second;
        for (int i = 0; i < item.numSubVectors; ++i) {
          newIterArgs.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
              loc, en.value(), item.offsets[i], targetShape, item.strides[i]));
        }
      }
    }

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block& newBlock = newForOp.getRegion().front();

    // 1, merge block args to restore the old vector type
    rewriter.setInsertionPointToStart(&newBlock);
    SmallVector<Value> newBlockArgs;
    newBlockArgs.push_back(newForOp.getInductionVar());
    // skip the first block arg: loop induction var.
    size_t blockArgIdx = 1;
    for (int i = 0; i < forOp->getNumResults(); ++i) {
      auto it = candidateValueMap.find(i);
      if (it == candidateValueMap.end()) {
        newBlockArgs.push_back(newBlock.getArgument(blockArgIdx++));
        continue;
      }
      auto& item = it->second;
      Value mergedVec = rewriter.create<arith::ConstantOp>(
          loc, item.dstVecType, rewriter.getZeroAttr(item.dstVecType));
      for (int subIdx = 0; subIdx < item.numSubVectors; ++subIdx) {
        mergedVec = rewriter.create<vector::InsertStridedSliceOp>(
            loc, newBlock.getArgument(blockArgIdx++), mergedVec,
            item.offsets[subIdx], item.strides[subIdx]);
      }
      newBlockArgs.push_back(mergedVec);
    }

    // 2, clone ops inside the region of old for op.
    Block& oldBlock = forOp.getRegion().front();
    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockArgs);
    auto oldYieldOp = &newBlock.back();
    rewriter.setInsertionPointAfter(oldYieldOp);

    // 3, split the yield result of old for op
    SmallVector<Value> newYieldValues;
    for (const auto& en : llvm::enumerate(oldYieldOp->getOperands())) {
      auto it = candidateValueMap.find(en.index());
      if (it == candidateValueMap.end()) {
        newYieldValues.push_back(en.value());
        continue;
      }
      auto& item = it->second;
      for (int subIdx = 0; subIdx < item.numSubVectors; ++subIdx) {
        newYieldValues.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
            loc, en.value(), item.offsets[subIdx], targetShape,
            item.strides[subIdx]));
      }
    }
    rewriter.create<scf::YieldOp>(loc, newYieldValues);
    rewriter.eraseOp(oldYieldOp);

    // 4, merge return value of new for op.
    rewriter.setInsertionPointAfter(newForOp);
    size_t resultIdx = 0;
    SmallVector<Value> newResults;
    for (const auto& en : llvm::enumerate(forOp->getResults())) {
      auto it = candidateValueMap.find(en.index());
      if (it == candidateValueMap.end()) {
        newResults.push_back(newForOp->getResult(resultIdx++));
        continue;
      }
      auto& item = it->second;
      Value mergedVec = rewriter.create<arith::ConstantOp>(
          loc, item.dstVecType, rewriter.getZeroAttr(item.dstVecType));
      for (int subIdx = 0; subIdx < item.numSubVectors; ++subIdx) {
        mergedVec = rewriter.create<vector::InsertStridedSliceOp>(
            loc, newForOp->getResult(resultIdx++), mergedVec,
            item.offsets[subIdx], item.strides[subIdx]);
      }
      newResults.push_back(mergedVec);
    }
    rewriter.replaceOp(forOp, newResults);
    return success();
  }

 private:
  vector::UnrollVectorOptions options_;
};

}  // namespace

void DecomposeVectorsOp::build(OpBuilder& builder, OperationState& result,
                               Value target, int64_t vectorSize) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      DecomposeVectorsOp::getVectorSizeAttrName(result.name),
      builder.getIntegerAttr(builder.getIntegerType(64), vectorSize));
  result.addTypes({});
}

DiagnosedSilenceableFailure DecomposeVectorsOp::applyToOne(
    Operation* target, SmallVectorImpl<Operation*>& results,
    transform::TransformState& state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  MLIRContext* ctx = getContext();
  RewritePatternSet patterns(ctx);
  // decompose outerproduct to bcast + fma ops.
  vector::populateVectorContractLoweringPatterns(patterns);

  vector::UnrollVectorOptions options;
  auto isTargetType = [&](Type ty) {
    auto castedTy = ty.dyn_cast<VectorType>();
    return castedTy && castedTy.getRank() > 0 &&
           castedTy.getShape()[castedTy.getRank() - 1] %
                   this->getVectorSize() ==
               0;
  };
  auto getVectorTypeOfForOp = [&](Operation* op) -> VectorType {
    if (!isa<scf::ForOp>(op)) return nullptr;
    VectorType vectorTy;
    for (auto ty : op->getResultTypes()) {
      if (!isTargetType(ty)) continue;
      if (vectorTy && vectorTy != ty.dyn_cast<VectorType>()) return nullptr;
      vectorTy = ty.dyn_cast<VectorType>();
    }
    return vectorTy;
  };
  options.setFilterConstraint([&](Operation* op) {
    if (getVectorTypeOfForOp(op)) return success();
    if (isa<vector::TransferReadOp, vector::TransferWriteOp>(op))
      return success();
    if (op->getNumResults() != 1) return failure();
    if (op->getDialect()->getTypeID() != TypeID::get<vector::VectorDialect>() &&
        op->getDialect()->getTypeID() != TypeID::get<arith::ArithDialect>())
      return failure();
    return success(isTargetType(op->getResult(0).getType()));
  });
  vector::UnrollVectorOptions::NativeShapeFnType nativeShapeFn =
      [&](Operation* op) -> Optional<SmallVector<int64_t, 4>> {
    VectorType targetVectorTy;
    if (auto vectorTy = getVectorTypeOfForOp(op)) {
      targetVectorTy = vectorTy;
    } else if (isa<vector::TransferWriteOp>(op)) {
      targetVectorTy = op->getOperand(0).getType().cast<VectorType>();
    } else {
      targetVectorTy = op->getResult(0).getType().cast<VectorType>();
    }
    SmallVector<int64_t, 4> nativeShape(targetVectorTy.getRank(), 1);
    nativeShape[targetVectorTy.getRank() - 1] = this->getVectorSize();
    return nativeShape;
  };
  options.setNativeShapeFn(nativeShapeFn);
  vector::populateVectorUnrollPatterns(patterns, options);
  patterns.insert<DecomposeVectorInOutsOfForOp>(ctx, options);

  // some clean up patterns.
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);

  // Apply everything.
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// LinalgFuseOperandOp
//===---------------------------------------------------------------------===//

void LinalgFuseOperandOp::build(OpBuilder& builder, OperationState& result,
                                Value target, int64_t operandIdx) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(
      LinalgFuseOperandOp::getOperandIdxAttrName(result.name),
      builder.getIntegerAttr(builder.getIntegerType(64), operandIdx));
  result.addTypes({pdl::OperationType::get(ctx)});
}

DiagnosedSilenceableFailure LinalgFuseOperandOp::applyToOne(
    Operation* target, SmallVectorImpl<Operation*>& results,
    transform::TransformState& state) {
  auto linalgOp = dyn_cast<linalg::GenericOp>(target);
  if (!linalgOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to linalg::GenericOp");
  }
  if (getOperandIdx() >= linalgOp->getNumOperands()) {
    return mlir::emitDefiniteFailure(target, "illegal operand_idx\n");
  }
  OpOperand& opOperand = linalgOp->getOpOperand(getOperandIdx());
  if (!linalg::areElementwiseOpsFusable(&opOperand)) {
    return mlir::emitDefiniteFailure(
        target, "operand of linalg::GenericOp is not fusible");
  }
  SimplePatternRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<Operation*> fusedOp =
      linalg::fuseElementwiseOps(rewriter, &opOperand);
  if (!succeeded(fusedOp)) {
    return mlir::emitDefiniteFailure(
        target, "failed to fuse operand into linalg::GenericOp");
  }
  // copy custom attributes.
  for (const auto& namedAttr : linalg::getPrunedAttributeList(linalgOp)) {
    (*fusedOp)->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  auto replacements =
      (*fusedOp)->getResults().take_back(linalgOp.getNumResults());
  rewriter.replaceOp(linalgOp, replacements);

  results.push_back(*fusedOp);
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// LinalgFuseProducersOp
//===---------------------------------------------------------------------===//

void LinalgFuseProducersOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::onlyReadsHandle(this->getOperands().drop_front(), effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure LinalgFuseProducersOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one target op but got ")
           << targetOps.size();
  Operation* targetOp = targetOps[0];
  auto linalgOp = dyn_cast<linalg::GenericOp>(targetOp);
  if (!linalgOp) {
    return mlir::emitDefiniteFailure(targetOp,
                                     "applies only to linalg::GenericOp");
  }

  // collect producers and assign an id for each producer.
  // The id system will help to get a deterministic fused result.
  DenseMap<Operation*, int> producerMap;
  for (Value producerHandle : getProducers()) {
    for (auto producer : state.getPayloadOps(producerHandle))
      if (producerMap.find(producer) == producerMap.end()) {
        producerMap[producer] = producerMap.size();
      }
  }

  bool stop = false;
  Operation* fusedOp = targetOp;
  SimplePatternRewriter rewriter(targetOp->getContext());
  rewriter.setInsertionPoint(targetOp);
  do {
    stop = true;
    int currentMaxIdx = -1;
    OpOperand* targetOpOperand = nullptr;
    for (auto& opOperand : fusedOp->getOpOperands()) {
      auto definingOp = opOperand.get().getDefiningOp();
      auto it = producerMap.find(definingOp);
      if (it == producerMap.end() ||
          !linalg::areElementwiseOpsFusable(&opOperand))
        continue;
      if (it->second > currentMaxIdx) {
        currentMaxIdx = it->second;
        targetOpOperand = &opOperand;
      }
    }

    if (!targetOpOperand) continue;
    FailureOr<Operation*> newFusedOp =
        linalg::fuseElementwiseOps(rewriter, targetOpOperand);
    if (!succeeded(newFusedOp)) {
      return mlir::emitDefiniteFailure(
          targetOp, "failed to fuse producer into linalg::GenericOp");
    }
    for (const auto& namedAttr : linalg::getPrunedAttributeList(linalgOp)) {
      (*newFusedOp)->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    fusedOp = *newFusedOp;
    stop = false;
  } while (!stop);

  if (fusedOp != targetOp) {
    auto replacements =
        fusedOp->getResults().take_back(linalgOp.getNumResults());
    rewriter.replaceOp(linalgOp, replacements);
  }

  results.set(getResult().cast<OpResult>(), {fusedOp});
  return DiagnosedSilenceableFailure(success());
}

}  // namespace transform_dialect

void registerTransformDialectCommonExtension(DialectRegistry& registry) {
  registry
      .addExtensions<::mlir::disc_ral::transform_dialect::CommonExtensions>();
}

}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.cc.inc"
