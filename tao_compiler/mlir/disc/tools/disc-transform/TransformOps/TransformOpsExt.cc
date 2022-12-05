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

#include "tensorflow/compiler/mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"

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
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/utils.h"

namespace mlir {
namespace disc_ral {
namespace transform_dialect {

CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.cc.inc"
      >();
}

//===---------------------------------------------------------------------===//
// DISCBufferizeOp
//===---------------------------------------------------------------------===//

using bufferization::BufferizationOptions;
using bufferization::OneShotAnalysisState;
using bufferization::OneShotBufferizationOptions;

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

  if (failed(pm.run(state.getTopLevel())))
    return DiagnosedSilenceableFailure::definiteFailure();

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

}  // namespace transform_dialect

void registerTransformDialectCommonExtension(DialectRegistry& registry) {
  registry
      .addExtensions< ::mlir::disc_ral::transform_dialect::CommonExtensions>();
}

}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.cc.inc"
