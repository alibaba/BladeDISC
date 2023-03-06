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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/utils.h"

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
  return success(createLinalgCopyOp(builder, loc, from, to) != nullptr);
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
      ::mlir::bufferization::LayoutMapOption::IdentityLayoutMap;

  // bufferization.to_memref is used to bufferize constant_wrapper ops. DISC has
  // it's own logic to handle constants. We'd like to leave the these constant
  // ops as is and insert bufferization.to_memref to convert the tensor to
  // memref.
  options.opFilter.denyOperation<disc_linalg_ext::ConstantWrapperOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
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
  /// 2, some tensor.empty can be folded. Try to do some optimizations in such
  /// case.
  IRRewriter rewriter(moduleOp->getContext());
  OneShotAnalysisState oneShotState(moduleOp, options);
  if (failed(bufferization::analyzeOp(moduleOp, oneShotState)) ||
      failed(bufferization::insertSliceAnchoredEmptyTensorEliminationStep(
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

bool allEquals(ArrayRef<int64_t> vals, int val) {
  return llvm::all_of(vals, [&](int64_t target) { return target == val; });
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
      if (std::get<1>(z) == ShapedType::kDynamic) return failure();
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

/// convert:
///   %0 = disc_linalg_ext.constant_wrapper dense<0.000000e+00> ...
///   %1 = tensor.extract %0 ...
///   use(%1)
/// to:
///   %0 = arith.constant 0 : float
///   use(%0)
struct FoldTensorExtractOfConstantWrapperPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto constOp = op.getTensor().getDefiningOp<ConstantWrapperOp>();
    if (!constOp) return failure();

    // Collect the constant indices into the tensor.
    SmallVector<uint64_t, 8> indices;
    for (Value indice : llvm::drop_begin(op->getOperands(), 1)) {
      auto constIndexOp = indice.getDefiningOp<arith::ConstantIndexOp>();
      if (!constIndexOp) return failure();
      indices.push_back(constIndexOp.getValue().cast<IntegerAttr>().getInt());
    }

    auto attr = constOp.getValue().getValues<Attribute>()[indices];
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getResult().getType(),
                                                   attr);
    return success();
  }
};

/// convert:
///   %0 = ... : vector<16xf32>
///   %1 = vector.transfer_write %0, %arg0[%c0, %c0]
///   %2 = disc_linalg_ext. dense<0.000000e+00> ...
///   %3 = vector.transfer_read %1[%c0, %c0], %2 ...
///   use(%3)
/// to:
///   %0 = ... : vector<16xf32>
///   use(%0)
struct FoldXferReadOfXferWriterPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter& rewriter) const override {
    auto writeOp = op.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!writeOp || writeOp.getVector().getType() != op->getResult(0).getType())
      return failure();

    // check all the indices are equal
    if (op.getIndices().size() != writeOp.getIndices().size()) return failure();
    for (auto [idx0, idx1] : llvm::zip(op.getIndices(), writeOp.getIndices()))
      if (idx0 != idx1) return failure();

    // check padding value of xfer read op is padding_value_placeholder with
    // kAny mode padding.
    auto placeholderOp =
        op.getPadding()
            .getDefiningOp<disc_linalg_ext::PaddingValuePlaceholderOp>();
    if (!placeholderOp ||
        placeholderOp.getMode() != disc_linalg_ext::PaddingValueModeEnum::kAny)
      return failure();

    rewriter.replaceOp(op, writeOp.getVector());
    return success();
  }
};

/// convert:
///   %0 = vector.transfer_read %0, %cst : vector<8x12xf32>
///   %1 = arith.select %pred, %0, %vector_cst : vector<8x12xf32>
///   %2 = vector.transfer_write %1, %0[...]
///   %3 = vector.transfer_read %0[...], %cst : vector<8x12xf32>
///   use(%3)
/// to:
///   %0 = vector.transfer_read %0, %cst : vector<8x12xf32>
///   %1 = arith.select %pred, %0, %vector_cst : vector<8x12xf32>
///   use(%1)
struct FoldXferReadOfXferWriterWithSelectPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter& rewriter) const override {
    // Find xfer write op
    auto writeOp = op.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!writeOp || writeOp.getVector().getType() != op->getResult(0).getType())
      return failure();

    // Find select op
    auto selectOp = writeOp.getVector().getDefiningOp<arith::SelectOp>();
    if (!selectOp ||
        selectOp->getResult(0).getType() != op->getResult(0).getType())
      return failure();

    // find const op and another xfer read op
    auto prevReadOp =
        selectOp->getOperand(1).getDefiningOp<vector::TransferReadOp>();
    auto vectorConstOp =
        selectOp->getOperand(2).getDefiningOp<arith::ConstantOp>();
    if (!vectorConstOp || !prevReadOp) {
      prevReadOp =
          selectOp->getOperand(2).getDefiningOp<vector::TransferReadOp>();
      vectorConstOp =
          selectOp->getOperand(1).getDefiningOp<arith::ConstantOp>();
    }
    if (!vectorConstOp || !prevReadOp) return failure();
    auto denseAttr = vectorConstOp.getValue().cast<DenseElementsAttr>();
    if (!denseAttr.isSplat()) return failure();

    // check the first read op and the second read have the same padding value
    if (op.getPadding() != prevReadOp.getPadding()) return failure();

    // check the padding value for xfer read op is a const
    auto scalarConstOp = op.getPadding().getDefiningOp<arith::ConstantOp>();
    if (!scalarConstOp) return failure();

    if (*denseAttr.getValues<Attribute>().begin() != scalarConstOp.getValue())
      return failure();

    rewriter.replaceOp(op, selectOp->getResults());
    return success();
  }
};

Operation* getAutomaticAllocationScope(Operation* op) {
  // Find the closest surrounding allocation scope that is not a known looping
  // construct (putting alloca's in loops doesn't always lower to deallocation
  // until the end of the loop).
  Operation* scope = nullptr;
  for (Operation* parent = op->getParentOp(); parent != nullptr;
       parent = parent->getParentOp()) {
    if (parent->hasTrait<OpTrait::AutomaticAllocationScope>()) scope = parent;
    if (scope && !isa<scf::ForOp, AffineForOp>(scope)) break;
  }
  return scope;
}

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
        (aShape[idx] == bShape[idx]) ? aShape[idx] : ShapedType::kDynamic;
    resStrides[idx] =
        (aStrides[idx] == bStrides[idx]) ? aStrides[idx] : ShapedType::kDynamic;
  }
  resOffset = (aOffset == bOffset) ? aOffset : ShapedType::kDynamic;
  return MemRefType::get(
      resShape, aT.getElementType(),
      StridedLayoutAttr::get(aT.getContext(), resOffset, resStrides));
}

/// convert:
///   %0 = vector.transfer_read %src, %cst : vector<8x12xf32>
///   %1 = arith.select %pred, %0, %vector_cst : vector<8x12xf32>
///   use(%1)
/// to:
///   %0 = memref.alloca() : memref<8x12xf32>
///   %buffer = scf.if %pred -> memref<?x?xf32> {
///     scf.yield %src
///   else {
///     scf.yield %0
///   }
///   %1 = vector.transfer_read %buffer, %cst : vector<8x12xf32>
///   use(%1)
///
/// Note that we need this pattern because we observed that lowered assemble
/// code of arith.select is very slow.
struct SelectOfXferReadAndConstToFillPattern
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter& rewriter) const override {
    auto vectorTy = selectOp->getResult(0).getType().dyn_cast<VectorType>();
    if (!vectorTy) return failure();

    // find const op and xfer read op
    auto xferReadOp =
        selectOp->getOperand(1).getDefiningOp<vector::TransferReadOp>();
    auto vectorConstOp =
        selectOp->getOperand(2).getDefiningOp<arith::ConstantOp>();
    bool thenIsConst = false;
    if (!vectorConstOp || !xferReadOp) {
      xferReadOp =
          selectOp->getOperand(2).getDefiningOp<vector::TransferReadOp>();
      vectorConstOp =
          selectOp->getOperand(1).getDefiningOp<arith::ConstantOp>();
      thenIsConst = true;
    }
    if (!vectorConstOp || !xferReadOp) return failure();

    // check type is compatible
    auto srcTy = xferReadOp.getSource().getType().dyn_cast<MemRefType>();
    if (!srcTy) return failure();
    auto memrefTy =
        MemRefType::get(vectorTy.getShape(), vectorTy.getElementType());
    auto castTy = getCastCompatibleMemRefType(srcTy, memrefTy);
    if (!castTy) return failure();

    auto denseAttr = vectorConstOp.getValue().cast<DenseElementsAttr>();
    if (!denseAttr.isSplat()) return failure();

    // check the padding value for xfer read op is a const
    auto scalarConstOp =
        xferReadOp.getPadding().getDefiningOp<arith::ConstantOp>();
    if (!scalarConstOp) return failure();

    if (*denseAttr.getValues<Attribute>().begin() != scalarConstOp.getValue())
      return failure();

    Operation* scope = getAutomaticAllocationScope(selectOp);
    if (!scope) return failure();

    Location loc = selectOp.getLoc();
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&scope->getRegion(0).front());
    Value newInitBuffer = rewriter.create<memref::AllocaOp>(
        loc, memrefTy, ValueRange{}, rewriter.getI64IntegerAttr(64));
    rewriter.create<linalg::FillOp>(loc, ValueRange{xferReadOp.getPadding()},
                                    ValueRange{newInitBuffer});

    rewriter.setInsertionPoint(selectOp.getOperation());
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{castTy},
                                           selectOp->getOperand(0), true);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    Value thenResult =
        rewriter
            .create<memref::CastOp>(
                loc, castTy,
                thenIsConst ? newInitBuffer : xferReadOp.getSource())
            ->getResult(0);
    rewriter.create<scf::YieldOp>(loc, thenResult);
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    Value elseResult =
        rewriter
            .create<memref::CastOp>(
                loc, castTy,
                !thenIsConst ? newInitBuffer : xferReadOp.getSource())
            ->getResult(0);
    rewriter.create<scf::YieldOp>(loc, elseResult);

    rewriter.setInsertionPointAfter(ifOp);
    IRMapping mapping;
    mapping.map(xferReadOp.getSource(), ifOp->getResult(0));
    auto clonedXferReadOp = rewriter.clone(*xferReadOp.getOperation(), mapping);

    rewriter.replaceOp(selectOp, clonedXferReadOp->getResults());
    return success();
  }
};

template <typename T>
SmallVector<int64_t> getVectorSliceOpOffset(T op) {
  SmallVector<int64_t> offsets;
  if (op->hasAttr("offsets")) {
    for (Attribute attr : op->template getAttrOfType<ArrayAttr>("offsets")) {
      offsets.push_back(attr.cast<IntegerAttr>().getInt());
    }
  } else {
    assert(op->hasAttr("position"));
    for (Attribute attr : op->template getAttrOfType<ArrayAttr>("position")) {
      offsets.push_back(attr.cast<IntegerAttr>().getInt());
    }
  }
  return offsets;
}

bool checkMajorDimsAllOnes(VectorType ty) {
  if (ty.getRank() <= 0) return false;
  for (int64_t d = 0; d < ty.getRank() - 1; ++d) {
    if (ty.getShape()[d] != 1) return false;
  }
  return true;
}

struct VectorExtractStridedSliceOfInsertStridedSliceFold
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern<vector::ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractStridedOp,
                                PatternRewriter& rewriter) const override {
    // check only trivial stirdes
    if (extractStridedOp.hasNonUnitStrides()) return failure();

    // check the result vector shape match pattern: vector<1x1x...xVecxelemType>
    auto vectorTy = extractStridedOp->getResult(0).getType().cast<VectorType>();
    if (!checkMajorDimsAllOnes(vectorTy)) return failure();

    // check the offset for the last dim is divisible by the result vector size.
    auto extractOffsets = getVectorSliceOpOffset(extractStridedOp);
    if (extractOffsets.back() % vectorTy.getShape()[vectorTy.getRank() - 1] !=
        0)
      return failure();

    Value targetValue;
    SmallVector<int64_t> targetOffsets = extractOffsets;
    Operation* sourceOp = extractStridedOp.getVector().getDefiningOp();
    while (sourceOp && !targetValue) {
      if (auto insertStridedOp =
              dyn_cast<vector::InsertStridedSliceOp>(sourceOp)) {
        if (insertStridedOp.hasNonUnitStrides()) return failure();
        if (targetOffsets == getVectorSliceOpOffset(insertStridedOp)) {
          auto ty = insertStridedOp.getSourceVectorType();
          if (!checkMajorDimsAllOnes(ty)) return failure();
          if (ty.getShape()[ty.getRank() - 1] !=
              vectorTy.getShape()[vectorTy.getRank() - 1])
            return failure();
          targetValue = insertStridedOp.getSource();
        } else {
          sourceOp = insertStridedOp.getDest().getDefiningOp();
        }
      } else if (auto insertOp = dyn_cast<vector::InsertOp>(sourceOp)) {
        bool match = true;
        auto insertOffsets = getVectorSliceOpOffset(insertOp);
        SmallVector<int64_t> newTargetOffsets;
        for (int64_t i = 0; match && i < targetOffsets.size(); ++i) {
          if (i < insertOffsets.size()) {
            match &= (insertOffsets[i] == targetOffsets[i]);
          } else {
            newTargetOffsets.push_back(targetOffsets[i]);
          }
        }

        if (match) {
          sourceOp = insertOp.getSource().getDefiningOp();
          std::swap(targetOffsets, newTargetOffsets);
        } else {
          sourceOp = insertOp.getDest().getDefiningOp();
        }
      } else if (auto extractOp = dyn_cast<vector::ExtractOp>(sourceOp)) {
        auto newTargetOffsets = getVectorSliceOpOffset(extractOp);
        newTargetOffsets.append(targetOffsets);
        sourceOp = extractOp.getVector().getDefiningOp();
        std::swap(newTargetOffsets, targetOffsets);
      } else {
        return failure();
      }
    }

    if (!targetValue) return failure();
    if (targetValue.getType() != vectorTy) {
      targetValue = rewriter.create<vector::ShapeCastOp>(
          extractStridedOp.getLoc(), vectorTy, targetValue);
    }
    rewriter.replaceOp(extractStridedOp, targetValue);

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
  patterns.insert<FoldTensorExtractOfConstantWrapperPattern>(ctx);
  patterns.insert<FoldXferReadOfXferWriterPattern>(ctx);
  patterns.insert<FoldXferReadOfXferWriterWithSelectPattern>(ctx);
  patterns.insert<SelectOfXferReadAndConstToFillPattern>(ctx);
  patterns.insert<VectorExtractStridedSliceOfInsertStridedSliceFold>(ctx);
  linalg::populateEraseUnnecessaryInputsPatterns(patterns);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
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
    Operation* target, transform::ApplyToEachResultList& results,
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

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
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
    Operation* target, transform::ApplyToEachResultList& results,
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

  results.push_back(sliceOp.getOperation());
  return DiagnosedSilenceableFailure::success();
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
  SmallVector<utils::IteratorType> iteratorTypes(transposeVector.size(),
                                                 utils::IteratorType::parallel);

  // Create a GenericOp to transpose `inputTensor` into `outputTensor`.
  auto transposeOp =
      b.create<linalg::GenericOp>(loc, resultTensorType, inputTensor,
                                  outputTensor, indexingMaps, iteratorTypes);
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
  return DiagnosedSilenceableFailure::success();
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
    Operation* target, transform::ApplyToEachResultList& results,
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
    if (staticStep == 1 || staticSrcDimUppers[srcIdx] != ShapedType::kDynamic &&
                               staticSrcDimUppers[srcIdx] % staticStep == 0) {
      srcDimUppers[srcIdx] = step;
      staticSrcDimUppers[srcIdx] = staticStep;
    } else {
      Value remaining = b.create<arith::SubIOp>(loc, upper, iv);
      srcDimUppers[srcIdx] = buildMin(b, loc, remaining, step);
      staticSrcDimUppers[srcIdx] = ShapedType::kDynamic;
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

  results.push_back(forOps[0]);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// InlineReductionInitializer
//===---------------------------------------------------------------------===//

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
  return DiagnosedSilenceableFailure::success();
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

struct DecomposeVectorInOutsOfIfOp : public OpRewritePattern<scf::IfOp> {
  explicit DecomposeVectorInOutsOfIfOp(
      MLIRContext* context, const vector::UnrollVectorOptions& options,
      PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<scf::IfOp>(context, benefit),
        options_(options) {}

  struct VectorDecomposeInfo {
    VectorType dstVecType;
    int64_t startIdx;
    int64_t numSubVectors;
    SmallVector<SmallVector<int64_t>> strides;
    SmallVector<SmallVector<int64_t>> offsets;
  };

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter& rewriter) const final {
    Location loc = ifOp.getLoc();
    if (failed(options_.filterConstraint(ifOp))) return failure();
    auto maybeTargetShape = options_.nativeShape(ifOp);
    if (!maybeTargetShape) return failure();
    auto& targetShape = *maybeTargetShape;

    int numNewOuts = 0;
    VectorType targetVectorType;
    DenseMap<int, VectorDecomposeInfo> candidateValueMap;
    SmallVector<Type> newResultTypes;
    for (const auto& en : llvm::enumerate(ifOp->getResults())) {
      auto ty = en.value().getType().dyn_cast<VectorType>();
      Optional<SmallVector<int64_t>> maybeShapeRatio;
      if (!ty ||
          !(maybeShapeRatio = computeShapeRatio(ty.getShape(), targetShape)) ||
          llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
        ++numNewOuts;
        newResultTypes.push_back(en.value().getType());
        continue;
      }
      // TODO(wyzero): support multiple iter args with different vector type.
      if (targetVectorType && targetVectorType != ty) return failure();
      targetVectorType = ty;
      auto& item = candidateValueMap[en.index()];
      item.dstVecType = ty;
      item.numSubVectors = computeMaxLinearIndex(*maybeShapeRatio);
      item.startIdx = numNewOuts;
      SmallVector<int64_t> strides(targetShape.size(), 1);
      for (int i = 0; i < item.numSubVectors; ++i) {
        auto offsets = getVectorOffset(ty.getShape(), targetShape, i);
        item.strides.push_back(strides);
        item.offsets.push_back(offsets);
        newResultTypes.push_back(
            VectorType::get(targetShape, ty.getElementType()));
        ++numNewOuts;
      }
    }

    if (candidateValueMap.empty() || ifOp.getElseRegion().empty())
      return failure();
    scf::IfOp newIfOp = rewriter.create<scf::IfOp>(loc, newResultTypes,
                                                   ifOp.getCondition(), true);

    auto cloneAndUpdateRegion = [&](Region& src, Region& dst) {
      dst.takeBody(src);
      SmallVector<Value> newResults;
      auto yieldOp = dst.front().getTerminator();
      rewriter.setInsertionPoint(yieldOp);
      for (auto [idx, val] : llvm::enumerate(yieldOp->getOperands())) {
        auto it = candidateValueMap.find(idx);
        if (it == candidateValueMap.end()) newResults.push_back(val);
        auto& item = it->second;
        for (int i = 0; i < item.numSubVectors; ++i) {
          newResults.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
              loc, val, item.offsets[i], targetShape, item.strides[i]));
        }
      }
      yieldOp->setOperands(newResults);
    };

    cloneAndUpdateRegion(ifOp.getThenRegion(), newIfOp.getThenRegion());
    cloneAndUpdateRegion(ifOp.getElseRegion(), newIfOp.getElseRegion());

    // merge the return values of new if op.
    rewriter.setInsertionPointAfter(newIfOp);
    size_t resultIdx = 0;
    SmallVector<Value> newResults;
    for (auto [idx, val] : llvm::enumerate(ifOp->getResults())) {
      auto it = candidateValueMap.find(idx);
      if (it == candidateValueMap.end()) {
        newResults.push_back(newIfOp->getResult(resultIdx++));
        continue;
      }
      auto& item = it->second;
      Value mergedVec = rewriter.create<arith::ConstantOp>(
          loc, item.dstVecType, rewriter.getZeroAttr(item.dstVecType));
      for (int subIdx = 0; subIdx < item.numSubVectors; ++subIdx) {
        mergedVec = rewriter.create<vector::InsertStridedSliceOp>(
            loc, newIfOp->getResult(resultIdx++), mergedVec,
            item.offsets[subIdx], item.strides[subIdx]);
      }
      newResults.push_back(mergedVec);
    }
    rewriter.replaceOp(ifOp, newResults);
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
    Operation* target, transform::ApplyToEachResultList& results,
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
  auto getVectorTypeOfForOrIfOp = [&](Operation* op) -> VectorType {
    if (!isa<scf::ForOp, scf::IfOp>(op)) return nullptr;
    VectorType vectorTy;
    for (auto ty : op->getResultTypes()) {
      if (!isTargetType(ty)) continue;
      if (vectorTy && vectorTy != ty.dyn_cast<VectorType>()) return nullptr;
      vectorTy = ty.dyn_cast<VectorType>();
    }
    return vectorTy;
  };
  options.setFilterConstraint([&](Operation* op) {
    if (getVectorTypeOfForOrIfOp(op)) return success();
    if (isa<vector::TransferReadOp, vector::TransferWriteOp>(op))
      return success();
    if (op->getNumResults() != 1) return failure();
    if (op->getDialect()->getTypeID() != TypeID::get<vector::VectorDialect>() &&
        op->getDialect()->getTypeID() != TypeID::get<arith::ArithDialect>() &&
        op->getDialect()->getTypeID() != TypeID::get<math::MathDialect>())
      return failure();
    return success(isTargetType(op->getResult(0).getType()));
  });
  vector::UnrollVectorOptions::NativeShapeFnType nativeShapeFn =
      [&](Operation* op) -> Optional<SmallVector<int64_t, 4>> {
    VectorType targetVectorTy;
    if (auto vectorTy = getVectorTypeOfForOrIfOp(op)) {
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
  patterns.insert<DecomposeVectorInOutsOfIfOp>(ctx, options);

  // some clean up patterns.
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);

  // Apply everything.
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
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
    Operation* target, transform::ApplyToEachResultList& results,
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
  FailureOr<linalg::ElementwiseOpFusionResult> fusedOp =
      linalg::fuseElementwiseOps(rewriter, &opOperand);
  if (!succeeded(fusedOp)) {
    return mlir::emitDefiniteFailure(
        target, "failed to fuse operand into linalg::GenericOp");
  }
  // copy custom attributes.
  for (const auto& namedAttr : linalg::getPrunedAttributeList(linalgOp)) {
    (*fusedOp).fusedOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  auto replacements =
      (*fusedOp).fusedOp->getResults().take_back(linalgOp.getNumResults());
  rewriter.replaceOp(linalgOp, replacements);

  results.push_back((*fusedOp).fusedOp);
  return DiagnosedSilenceableFailure::success();
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
    FailureOr<linalg::ElementwiseOpFusionResult> newFusedOp =
        linalg::fuseElementwiseOps(rewriter, targetOpOperand);
    if (!succeeded(newFusedOp)) {
      return mlir::emitDefiniteFailure(
          targetOp, "failed to fuse producer into linalg::GenericOp");
    }
    for (const auto& namedAttr : linalg::getPrunedAttributeList(linalgOp)) {
      (*newFusedOp).fusedOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    fusedOp = (*newFusedOp).fusedOp;
    stop = false;
  } while (!stop);

  if (fusedOp != targetOp) {
    auto replacements =
        fusedOp->getResults().take_back(linalgOp.getNumResults());
    rewriter.replaceOp(linalgOp, replacements);
  }

  results.set(getResult().cast<OpResult>(), {fusedOp});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ReplaceConstPaddingValueOp
//===---------------------------------------------------------------------===//

void ReplaceConstPaddingValueOp::build(OpBuilder& builder,
                                       OperationState& result, Value target,
                                       StringRef mode) {
  MLIRContext* ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(ReplaceConstPaddingValueOp::getModeAttrName(result.name),
                      builder.getStringAttr(mode));
  result.addTypes({pdl::OperationType::get(ctx)});
}

DiagnosedSilenceableFailure ReplaceConstPaddingValueOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto padOp = dyn_cast<tensor::PadOp>(target);
  if (!padOp) {
    return mlir::emitDefiniteFailure(target, "applies only to tensor::PadOp");
  }

  auto mode =
      disc_linalg_ext::symbolizeEnum<disc_linalg_ext::PaddingValueModeEnum>(
          getMode());
  if (!mode) {
    return mlir::emitDefiniteFailure(target, "invalid mode attr");
  }

  OpBuilder b(target);
  auto yieldOp = padOp.getBody()->getTerminator();
  Attribute value;
  if (!matchPattern(yieldOp->getOperand(0), m_Constant(&value))) {
    return mlir::emitDefiniteFailure(target, "applies only to pad with const");
  }
  auto placeholder = b.create<disc_linalg_ext::PaddingValuePlaceholderOp>(
      padOp.getLoc(), value, *mode);
  yieldOp->replaceUsesOfWith(yieldOp->getOperand(0), placeholder->getResult(0));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ConvertPaddingPlaceholderToConstOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure ConvertPaddingPlaceholderToConstOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto placeholderOp =
      dyn_cast<disc_linalg_ext::PaddingValuePlaceholderOp>(target);
  if (!placeholderOp) {
    return mlir::emitDefiniteFailure(
        target, "applies only to PaddingValuePlaceholderOp");
  }

  OpBuilder b(target);

  auto constOp =
      b.create<arith::ConstantOp>(target->getLoc(), placeholderOp.getValue());
  target->replaceAllUsesWith(constOp->getResults());
  results.push_back(constOp.getOperation());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// LinalgEagerlyBackwardInitTensorOp
//===---------------------------------------------------------------------===//

namespace {

OpOperand* backwardToFindValidValueToToReuse(OpOperand* v) {
  if (!v || !llvm::hasSingleElement(v->get().getUsers())) return nullptr;
  auto definingOp = v->get().getDefiningOp<linalg::LinalgOp>();
  if (!definingOp) return nullptr;
  int64_t resultIdx = v->get().cast<OpResult>().getResultNumber();
  auto initOperand = definingOp.getDpsInitOperand(resultIdx);
  if (!definingOp.payloadUsesValueFromOperand(initOperand)) return initOperand;
  return backwardToFindValidValueToToReuse(initOperand);
}

}  // namespace

DiagnosedSilenceableFailure LinalgEagerlyBackwardInitTensorOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
  if (!linalgOp)
    return mlir::emitDefiniteFailure(target, "applies only to linalgOp");

  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return mlir::emitDefiniteFailure(target,
                                     "not support reduction loop a.t.m.");

  if (linalgOp.getNumDpsInits() != 1)
    return mlir::emitDefiniteFailure(target,
                                     "not support multi outputs a.t.m.");

  auto initOperand = linalgOp.getDpsInitOperand(0);
  auto affineMap = linalgOp.getMatchingIndexingMap(initOperand);
  if (!affineMap.isPermutation())
    return mlir::emitDefiniteFailure(
        target, "only support permutation mapping for result");
  if (linalgOp.payloadUsesValueFromOperand(initOperand))
    return mlir::emitDefiniteFailure(
        target, "Not support if the result is used by the payload.");

  OpOperand* candidateOperand{nullptr};
  OpOperand* rootCandidateOperand{nullptr};
  for (auto operand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.getMatchingIndexingMap(operand) != affineMap) continue;
    candidateOperand = operand;
    rootCandidateOperand = backwardToFindValidValueToToReuse(candidateOperand);
    if (rootCandidateOperand) break;
  }

  // Early return if no valid candidates.
  if (!rootCandidateOperand) {
    results.push_back({target});
    return DiagnosedSilenceableFailure::success();
  }

  OpBuilder b(target);
  rootCandidateOperand->getOwner()->replaceUsesOfWith(
      rootCandidateOperand->get(), initOperand->get());
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<Value> newOutputs{candidateOperand->get()};
  for (auto operand : linalgOp.getDpsInputOperands()) {
    if (operand == candidateOperand) continue;
    newInputs.push_back(operand->get());
    newIndexingMaps.push_back(linalgOp.getMatchingIndexingMap(operand));
  }
  // for the new init operand
  newIndexingMaps.push_back(affineMap);
  auto iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                      ? TypeRange(ValueRange(newOutputs))
                                      : TypeRange{};
  SmallVector<NamedAttribute> attrs;
  for (auto attr : linalgOp->getAttrs()) {
    if (attr.getName().getValue().starts_with("disc"))
      attrs.push_back(std::move(attr));
  }

  auto bodyBuilder = [&](OpBuilder& innerBuilder, Location loc,
                         ValueRange operands) {
    IRMapping mapping;
    int64_t operandIdx = candidateOperand->getOperandNumber();
    if (operandIdx > 0)
      mapping.map(linalgOp.getBlock()->getArguments().take_front(operandIdx),
                  operands.take_front(operandIdx));
    if (operandIdx + 1 < linalgOp.getBlock()->getArguments().size())
      mapping.map(
          linalgOp.getBlock()->getArguments().drop_front(operandIdx + 1),
          operands.drop_front(operandIdx));
    mapping.map(linalgOp.getBlock()->getArgument(operandIdx), operands.back());
    for (Operation& op : *linalgOp.getBlock()) {
      innerBuilder.clone(op, mapping);
    }
  };

  auto genericOp = b.create<linalg::GenericOp>(
      linalgOp.getLoc(), resultTypes, newInputs, newOutputs, newIndexingMaps,
      iterators, bodyBuilder, attrs);
  target->replaceAllUsesWith(genericOp->getResults());
  results.push_back({genericOp.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DISCFuseIntoContainingOp
//===----------------------------------------------------------------------===//

namespace {

// First, find the first "scf::ForeachThreadOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation* tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase& rewriter, Diagnostic& diag, Operation* producerOp,
    Operation* containingOp) {
  LLVM_DEBUG(
      llvm::dbgs() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
  }

  // Search the first use by a "scf::ForOp" user.
  scf::ForOp forOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand& use) {
        forOp = dyn_cast<scf::ForOp>(use.getOwner());
        return forOp;
      });
  // If it's not from the containing op, return.
  if (!forOp || forOp != containingOp) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find a use by the containing op: " << *tileableProducer;
    return nullptr;
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand* pUse = &(*itProducerUses);
  BlockArgument bbArg = forOp.getBody()->getArgument(
      pUse->getOperandNumber() - forOp.getNumControlOperands() +
      forOp.getNumInductionVars());

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation* user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    diag.attachNote(containingOp->getLoc())
        << "could not find fusion opportunity for bbArg: " << bbArg;
    return nullptr;
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = pUse->get().cast<OpResult>().getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return nullptr;
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<Value> tiledProducer =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledProducer << "\n");

  // Replace the extract op.
  Operation* fusedOp = tiledProducer->getDefiningOp();
  rewriter.replaceOp(sliceOpToTile, fusedOp->getResult(resultNumber));

  // Replace the use in containingOp.
  rewriter.updateRootInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors[resultNumber]);
  });

  return fusedOp;
}

}  // namespace

void DISCFuseIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::onlyReadsHandle(this->getOperands().drop_front(), effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure DISCFuseIntoContainingOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  SmallVector<Operation*> fusedOps;
  ArrayRef<Operation*> producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(getFusedOp().cast<OpResult>(), SmallVector<mlir::Operation*>{});
    return DiagnosedSilenceableFailure::success();
  }
  ArrayRef<Operation*> containingOps = state.getPayloadOps(getContainingOp());
  if (containingOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << containingOps.size() << ")";
  }
  Operation* containingOp = containingOps.front();

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  SmallVector<Operation*> remainingProducers(producerOps.begin(),
                                             producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation*> {
    for (const auto& it : enumerate(remainingProducers)) {
      Operation* producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp = llvm::count_if(
          producerOp->getUsers(),
          [&](Operation* op) { return containingOp->isAncestor(op); });
      // TODO: When resolving the TODO below (no duplicate ops), take an op
      // that has no use among the remaining producers. This is a topological
      // sorting.
      if (numUsesInContainingOp > 0) {
        if (numUsesInContainingOp == 1)
          remainingProducers.erase(remainingProducers.begin() + it.index());
        return producerOp;
      }
    }
    return failure();
  };

  IRRewriter rewriter(getContext());
  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation*>());
      Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
      diag << "could not find next producer to fuse into container";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    Operation* producerOp = *nextProducer;

    // Default diagnostic, to be complemented with more failure information.
    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << " into " << *containingOp;

    Operation* tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, producerOp, containingOp);
    if (tiledContainingOpOperand) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\nFused an extract use through block argument\n"
                 << *containingOp);
      fusedOps.push_back(tiledContainingOpOperand);
      continue;
    }

    results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation*>());
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  results.set(getFusedOp().cast<OpResult>(), fusedOps);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ReductionOutputFuseOp
//===---------------------------------------------------------------------===//

namespace {

Operation* convertGenericOpToConditionalGenericOp(OpBuilder& b, Value pred,
                                                  linalg::LinalgOp linalgOp) {
  SmallVector<Value> inputs{pred};
  inputs.append(linalgOp.getDpsInputOperands());
  SmallVector<Value> outputs = linalgOp.getDpsInitOperands();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  if (indexingMaps.empty()) return nullptr;
  SmallVector<AffineMap> newIndexingMaps;
  newIndexingMaps.push_back(AffineMap::get(indexingMaps.back().getNumDims(),
                                           indexingMaps[0].getNumSymbols(), {},
                                           b.getContext()));
  newIndexingMaps.append(std::move(indexingMaps));
  auto iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};
  SmallVector<NamedAttribute> attrs;
  for (auto attr : linalgOp->getAttrs()) {
    if (attr.getName().getValue().starts_with("disc"))
      attrs.push_back(std::move(attr));
  }
  auto bodyBuilder = [&](OpBuilder& innerBuilder, Location loc,
                         ValueRange operands) {
    IRMapping mapping;
    mapping.map(linalgOp.getBlock()->getArguments(), operands.drop_front());
    for (Operation& op : linalgOp.getBlock()->without_terminator()) {
      innerBuilder.clone(op, mapping);
    }
    SmallVector<Value> newResults;
    for (Value v : linalgOp.getBlock()->getTerminator()->getOperands()) {
      newResults.push_back(mapping.lookupOrDefault(v));
    }
    innerBuilder.create<disc_linalg_ext::YieldOp>(
        linalgOp.getBlock()->getTerminator()->getLoc(), newResults);
  };
  auto conditionalGenericOp = b.create<disc_linalg_ext::ConditionalGenericOp>(
      linalgOp.getLoc(), resultTypes, inputs, outputs, newIndexingMaps,
      iterators, bodyBuilder, attrs);
  return conditionalGenericOp;
}

}  // namespace

void ReductionOutputFuseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::consumesHandle({this->getOperation()->getOperand(1)}, effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure ReductionOutputFuseOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one target but got ")
           << targetOps.size();
  Operation* target = targetOps[0];
  auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
  if (!linalgOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the target is a linalg op");

  ArrayRef<Operation*> loopOps = state.getPayloadOps(getLoop());
  if (loopOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one loop but got ")
           << loopOps.size();
  auto forOp = dyn_cast<scf::ForOp>(loopOps[0]);
  if (!forOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the loop to a for op");

  // Check all the outputs of the linalg have the same affine map.
  if (linalgOp->getNumResults() < 1)
    return mlir::emitDefiniteFailure(
        this->getOperation(), "expect the target has at least one result");
  auto operands = linalgOp.getDpsInitOperands();
  auto indexMap = linalgOp.getMatchingIndexingMap(operands.front());
  AffineMap resultIndexMap;
  for (auto [idx, operand] : llvm::enumerate(linalgOp.getDpsInitOperands())) {
    auto indexMap = linalgOp.getMatchingIndexingMap(operand);
    if (idx == 0) {
      resultIndexMap = indexMap;
    } else if (resultIndexMap != indexMap) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "expect all the results of the target have the same shape");
    }
  }
  // result of for op -> result idx.
  DenseMap<Value, int> forOpResultMap;
  for (auto [idx, result] : llvm::enumerate(forOp->getResults())) {
    forOpResultMap[result] = idx;
  }
  // linalg input operand idx -> result idx of for op.
  DenseMap<int, int> linalgInputOperandToForResultMap;
  for (auto [idx, operand] : llvm::enumerate(linalgOp.getDpsInputOperands())) {
    auto it = forOpResultMap.find(operand->get());
    if (it == forOpResultMap.end()) continue;
    auto indexMap = linalgOp.getMatchingIndexingMap(operand);
    if (resultIndexMap != indexMap) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "expect all the results of for loop which are consumed by the target "
          "have the same shape as the results of linalg op");
    }
    linalgInputOperandToForResultMap[idx] = it->second;
  }
  // linalg init operand idx -> result idx of for op.
  DenseMap<int, int> linalgInitOperandToForResultMap;
  DenseMap<int, int> forResultToLinalgInitOperandMap;
  for (auto [idx, operand] : llvm::enumerate(linalgOp.getDpsInitOperands())) {
    auto it = forOpResultMap.find(operand->get());
    if (it == forOpResultMap.end()) continue;
    auto indexMap = linalgOp.getMatchingIndexingMap(operand);
    if (resultIndexMap != indexMap) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "expect all the results of for loop which are consumed by the target "
          "have the same shape as the results of linalg op");
    }
    if (!llvm::hasSingleElement(operand->get().getUsers())) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "expect the init operand of target which is produced by the loop op "
          "only has one consumer");
    }
    linalgInitOperandToForResultMap[idx] = it->second;
    forResultToLinalgInitOperandMap[it->second] = idx;
  }

  if (linalgInputOperandToForResultMap.empty() &&
      linalgInitOperandToForResultMap.empty()) {
    return mlir::emitDefiniteFailure(
        this->getOperation(),
        "none of the results of the reduction loop are consumed by target\n");
  }

  OpBuilder b(target);
  auto newInitArgs = llvm::to_vector<>(forOp.getInitArgs());
  for (auto [idx, operand] : llvm::enumerate(linalgOp.getDpsInitOperands())) {
    if (linalgInitOperandToForResultMap.find(idx) !=
        linalgInitOperandToForResultMap.end())
      continue;
    newInitArgs.push_back(operand->get());
  }
  auto newForOp =
      b.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [oldValue, newValue] :
       llvm::zip(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
    mapping.map(oldValue, newValue);
  }
  int nonFusedInitOperandIdx = 0;
  for (auto [idx, operand] : llvm::enumerate(linalgOp.getDpsInitOperands())) {
    auto it = linalgInitOperandToForResultMap.find(idx);
    if (linalgInitOperandToForResultMap.find(idx) !=
        linalgInitOperandToForResultMap.end())
      continue;
    mapping.map(operand->get(),
                newForOp.getRegionIterArgs()
                    .drop_front(forOp.getRegionIterArgs().size() +
                                nonFusedInitOperandIdx++)
                    .front());
  }

  // move ops inside the old for op to the region of the new for op.
  for (Value arg : forOp.getBody()->getArguments()) {
    arg.replaceAllUsesWith(mapping.lookup(arg));
  }
  for (auto& op : llvm::make_early_inc_range(*forOp.getBody())) {
    op.moveBefore(newForOp.getBody(), newForOp.getBody()->end());
  }

  int forResultIdx;
  if (!linalgInputOperandToForResultMap.empty()) {
    forResultIdx = linalgInputOperandToForResultMap.begin()->second;
  } else if (!linalgInitOperandToForResultMap.empty()) {
    forResultIdx = linalgInitOperandToForResultMap.begin()->second;
  }

  auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  auto insertSliceOp =
      yieldOp->getOperand(forResultIdx).getDefiningOp<tensor::InsertSliceOp>();
  b.setInsertionPoint(yieldOp);
  for (auto [idx, val] : llvm::enumerate(yieldOp->getOperands())) {
    auto it = forResultToLinalgInitOperandMap.find(idx);
    if (it == forResultToLinalgInitOperandMap.end()) {
      mapping.map(forOp->getResult(idx), val);
      continue;
    }
    if (auto insertOp = val.getDefiningOp<tensor::InsertSliceOp>()) {
      mapping.map(forOp->getResult(idx), insertOp->getOperand(0));
    } else {
      mapping.map(forOp->getResult(idx), val);
    }
  }
  auto clonedTarget = b.clone(*target, mapping);
  Operation* tiledOp = clonedTarget;
  if (insertSliceOp) {
    auto tileableTarget = cast<TilingInterface>(clonedTarget);
    FailureOr<Value> tiledTarget = tileableTarget.generateResultTileValue(
        b, 0, insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes());
    if (failed(tiledTarget)) {
      return mlir::emitDefiniteFailure(this->getOperation(),
                                       "failed to tile target op\n");
    }
    tiledOp = tiledTarget->getDefiningOp();
  }
  Value iv = newForOp.getInductionVar();
  Value nextIv =
      b.create<arith::AddIOp>(tiledOp->getLoc(), iv, newForOp.getStep());
  Value pred =
      b.create<arith::CmpIOp>(tiledOp->getLoc(), arith::CmpIPredicate::sge,
                              nextIv, newForOp.getUpperBound());
  auto conditionalGenericOp = convertGenericOpToConditionalGenericOp(
      b, pred, cast<linalg::LinalgOp>(tiledOp));
  if (!conditionalGenericOp) {
    return mlir::emitDefiniteFailure(
        this->getOperation(), "failed to build conditional_generic op\n");
  }
  SmallVector<Value> newResults;
  for (auto [idx, val] : llvm::enumerate(yieldOp->getOperands())) {
    auto it = forResultToLinalgInitOperandMap.find(idx);
    if (it == forResultToLinalgInitOperandMap.end()) {
      newResults.push_back(val);
      continue;
    }
    Value updateValue = conditionalGenericOp->getResult(it->second);
    if (!insertSliceOp) {
      newResults.push_back(updateValue);
      continue;
    }
    Value initValue = newForOp.getRegionIterArg(idx);
    newResults.push_back(b.create<tensor::InsertSliceOp>(
                              tiledOp->getLoc(), updateValue, initValue,
                              insertSliceOp.getMixedOffsets(),
                              insertSliceOp.getMixedSizes(),
                              insertSliceOp.getMixedStrides())
                             ->getResult(0));
  }
  nonFusedInitOperandIdx = 0;
  for (auto [idx, updateValue] :
       llvm::enumerate(conditionalGenericOp->getResults())) {
    auto it = linalgInitOperandToForResultMap.find(idx);
    if (it != linalgInitOperandToForResultMap.end()) continue;
    if (!insertSliceOp) {
      newResults.push_back(updateValue);
      continue;
    }
    Value initValue = newForOp.getRegionIterArgs()
                          .drop_front(forOp.getRegionIterArgs().size() +
                                      nonFusedInitOperandIdx++)
                          .front();
    newResults.push_back(b.create<tensor::InsertSliceOp>(
                              tiledOp->getLoc(), updateValue, initValue,
                              insertSliceOp.getMixedOffsets(),
                              insertSliceOp.getMixedSizes(),
                              insertSliceOp.getMixedStrides())
                             ->getResult(0));
  }
  b.create<scf::YieldOp>(yieldOp->getLoc(), newResults);
  yieldOp->erase();

  nonFusedInitOperandIdx = 0;
  for (auto [idx, val] : llvm::enumerate(linalgOp->getResults())) {
    Value newVal;
    auto it = linalgInitOperandToForResultMap.find(idx);
    if (it != linalgInitOperandToForResultMap.end()) {
      newVal = newForOp->getResult(it->second);
    } else {
      newVal = newForOp->getResult(forOp->getNumResults() +
                                   nonFusedInitOperandIdx++);
    }
    val.replaceAllUsesWith(newVal);
  }
  linalgOp->erase();
  forOp->replaceAllUsesWith(
      newForOp->getResults().take_front(forOp->getNumResults()));
  results.set(getTiledTarget().cast<OpResult>(), {conditionalGenericOp});
  results.set(getFusedLoop().cast<OpResult>(), {newForOp});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// ReductionInputFuseOp
//===---------------------------------------------------------------------===//

void ReductionInputFuseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle({this->getOperation()->getOperand(0)}, effects);
  transform::consumesHandle({this->getOperation()->getOperand(1)}, effects);
  transform::producesHandle(this->getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure ReductionInputFuseOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one target but got ")
           << targetOps.size();
  Operation* target = targetOps[0];
  auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
  if (!linalgOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the target is a linalg op");

  ArrayRef<Operation*> loopOps = state.getPayloadOps(getLoop());
  if (loopOps.size() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect only one loop but got ")
           << loopOps.size();
  auto forOp = dyn_cast<scf::ForOp>(loopOps[0]);
  if (!forOp)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the loop to a for op");

  if (linalgOp->getNumResults() != 1)
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "expect the target has one result");

  auto iterOperands = forOp.getIterOperands();
  auto argIt = llvm::find(iterOperands, linalgOp->getResult(0));
  if (argIt == iterOperands.end())
    return mlir::emitDefiniteFailure(
        this->getOperation(),
        "expect the result of target is the iter arg of the loop.");
  Value iterArg = forOp.getRegionIterArg(argIt - iterOperands.begin());
  tensor::ExtractSliceOp sliceOp;
  auto numCandidateSliceOp =
      llvm::count_if(iterArg.getUsers(), [&](Operation* user) -> bool {
        if (!forOp->isProperAncestor(user)) return false;
        if (auto candidate = dyn_cast<tensor::ExtractSliceOp>(user)) {
          return sliceOp = candidate;
        }
        return false;
      });
  if (numCandidateSliceOp > 1) {
    return mlir::emitDefiniteFailure(
        this->getOperation(),
        (Twine("only support at most one candidate extract_slice "
               "of target in the loop but got") +
         Twine(numCandidateSliceOp))
            .str());
  }

  OpBuilder b(target->getContext());
  if (sliceOp)
    b.setInsertionPoint(sliceOp);
  else
    b.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

  forOp->setOperand(
      forOp.getNumControlOperands() + (argIt - iterOperands.begin()),
      linalgOp.getDpsInitOperands().front()->get());
  auto clonedTarget = b.clone(*target);
  clonedTarget->replaceUsesOfWith(linalgOp.getDpsInitOperands().front()->get(),
                                  iterArg);
  Operation* tiledOp = clonedTarget;
  if (sliceOp) {
    auto tileableTarget = cast<TilingInterface>(clonedTarget);
    FailureOr<Value> tiledTarget = tileableTarget.generateResultTileValue(
        b, 0, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes());
    if (failed(tiledTarget)) {
      return mlir::emitDefiniteFailure(this->getOperation(),
                                       "failed to tile target op\n");
    }
    tiledOp = tiledTarget->getDefiningOp();
  }
  Value pred =
      b.create<arith::CmpIOp>(tiledOp->getLoc(), arith::CmpIPredicate::eq,
                              forOp.getInductionVar(), forOp.getLowerBound());
  auto conditionalGenericOp = convertGenericOpToConditionalGenericOp(
      b, pred, cast<linalg::LinalgOp>(tiledOp));
  if (!conditionalGenericOp) {
    return mlir::emitDefiniteFailure(
        this->getOperation(), "failed to build conditional_generic op\n");
  }
  if (sliceOp) {
    sliceOp->replaceAllUsesWith(conditionalGenericOp->getResults());
  } else {
    iterArg.replaceAllUsesExcept(conditionalGenericOp->getResult(0),
                                 {conditionalGenericOp});
  }
  results.set(getTiledTarget().cast<OpResult>(), {conditionalGenericOp});
  results.set(getFusedLoop().cast<OpResult>(), {forOp});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// VectorizeConditionalGenericOp
//===---------------------------------------------------------------------===//

namespace {

Operation* convertConditionalGenericOpToGenericOp(OpBuilder& b,
                                                  linalg::LinalgOp linalgOp) {
  SmallVector<Value> inputs = linalgOp.getDpsInputOperands();
  inputs.erase(inputs.begin());
  SmallVector<Value> outputs = linalgOp.getDpsInitOperands();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  indexingMaps.erase(indexingMaps.begin());
  auto iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};
  SmallVector<NamedAttribute> attrs;
  for (auto attr : linalgOp->getAttrs()) {
    if (attr.getName().getValue().starts_with("disc"))
      attrs.push_back(std::move(attr));
  }
  auto bodyBuilder = [&](OpBuilder& innerBuilder, Location loc,
                         ValueRange operands) {
    IRMapping mapping;
    mapping.map(linalgOp.getBlock()->getArguments().drop_front(), operands);
    for (Operation& op : linalgOp.getBlock()->without_terminator()) {
      innerBuilder.clone(op, mapping);
    }
    SmallVector<Value> newResults;
    for (Value v : linalgOp.getBlock()->getTerminator()->getOperands()) {
      newResults.push_back(mapping.lookupOrDefault(v));
    }
    innerBuilder.create<linalg::YieldOp>(
        linalgOp.getBlock()->getTerminator()->getLoc(), newResults);
  };
  auto genericOp = b.create<linalg::GenericOp>(linalgOp.getLoc(), resultTypes,
                                               inputs, outputs, indexingMaps,
                                               iterators, bodyBuilder, attrs);
  return genericOp;
}

}  // namespace

DiagnosedSilenceableFailure VectorizeConditionalGenericOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto genericOp = dyn_cast<disc_linalg_ext::ConditionalGenericOp>(target);
  if (!genericOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to ConditionalGenericOp");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  Location loc = genericOp->getLoc();
  Value pred = genericOp->getOperand(0);
  auto linalgOp = cast<linalg::LinalgOp>(target);

  // 0, check and collect init tensor types
  SmallVector<Type> resultTensorTypes;
  for (auto init : linalgOp.getDpsInitOperands()) {
    auto ty = init->get().getType().dyn_cast<RankedTensorType>();
    if (!ty || !ty.hasStaticShape()) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "expect the result of target op has ranked tensor type with static "
          "shape\n");
    }
    resultTensorTypes.push_back(ty);
  }

  // 0, build init loop like following:
  // ```
  //   scf.if %pred {
  //     disc_linalg_ext.conditional_generic ...
  //   } else {
  //     // simply forward init operands to outputs
  //     linalg.generic ...
  //   }
  // ```
  rewriter.setInsertionPoint(target);
  auto ifOp = rewriter.create<scf::IfOp>(loc, resultTensorTypes, pred, true);

  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  auto thenTarget = dyn_cast<linalg::LinalgOp>(
      convertConditionalGenericOpToGenericOp(rewriter, linalgOp));
  rewriter.create<scf::YieldOp>(loc, thenTarget->getResults());
  rewriter.setInsertionPoint(thenTarget);
  if (failed(linalg::vectorize(rewriter, thenTarget)))
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "failed to vectorize the then target\n");

  rewriter.setInsertionPointToStart(ifOp.elseBlock());
  auto elseTarget = dyn_cast<linalg::LinalgOp>(
      convertConditionalGenericOpToGenericOp(rewriter, linalgOp));
  elseTarget.getBlock()->getTerminator()->setOperands(
      elseTarget.getRegionOutputArgs());
  SmallVector<Operation*> toDeleteOps;
  for (auto& op : llvm::reverse(elseTarget.getBlock()->without_terminator())) {
    toDeleteOps.push_back(&op);
  }
  for (Operation* op : toDeleteOps) op->erase();
  // simply forward init input tensors to corresponding outputs in the else
  // branch.
  rewriter.create<scf::YieldOp>(loc, elseTarget->getResults());
  rewriter.setInsertionPoint(elseTarget);
  if (failed(linalg::vectorize(rewriter, elseTarget)))
    return mlir::emitDefiniteFailure(this->getOperation(),
                                     "failed to vectorize the else target\n");

  rewriter.setInsertionPointAfter(ifOp);

  // 1, replace the return type of ifOp: from tensor type to vector type
  SmallVector<Type> resultVectorTypes;
  SmallVector<Value> resultVectorValues;
  DenseMap<int, Operation*> resultIdx2writeOpMap;
  DenseMap<Operation*, int> writeOp2resultIdxMap;
  for (auto [idx, val] :
       llvm::enumerate(ifOp.elseBlock()->getTerminator()->getOperands())) {
    auto xferWriteOp = val.getDefiningOp<vector::TransferWriteOp>();
    if (!xferWriteOp) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "failed to find xfer write op after vectorization\n");
    }
    resultVectorValues.push_back(xferWriteOp.getVector());
    resultVectorTypes.push_back(xferWriteOp.getVector().getType());
    resultIdx2writeOpMap[idx] = xferWriteOp;
    writeOp2resultIdxMap[xferWriteOp] = idx;
  }

  auto vectorIfOp =
      rewriter.create<scf::IfOp>(loc, resultVectorTypes, pred, true);
  DenseMap<Value, Value> initTensor2LoadedVectorMap;
  Operation* moveAfterAnchor = vectorIfOp;
  for (auto& op :
       llvm::make_early_inc_range(ifOp.elseBlock()->without_terminator())) {
    if (auto xferWriteOp = dyn_cast<vector::TransferWriteOp>(&op)) {
      op.moveAfter(moveAfterAnchor);
      op.replaceUsesOfWith(xferWriteOp.getVector(),
                           vectorIfOp->getResult(writeOp2resultIdxMap[&op]));
      moveAfterAnchor = &op;
    } else {
      auto initArgs = linalgOp.getDpsInitOperands();
      auto xferReadOp = dyn_cast<vector::TransferReadOp>(&op);
      bool readFromInitArg =
          xferReadOp && llvm::find_if(initArgs, [&](OpOperand* operand) {
                          return operand->get() == xferReadOp.getSource();
                        }) != initArgs.end();
      if (!xferReadOp || readFromInitArg) op.moveBefore(ifOp);
      if (readFromInitArg) {
        initTensor2LoadedVectorMap[xferReadOp.getSource()] =
            xferReadOp->getResult(0);
      }
    }
  }
  rewriter.setInsertionPointToStart(vectorIfOp.elseBlock());
  rewriter.create<scf::YieldOp>(loc, resultVectorValues);
  rewriter.setInsertionPointToStart(vectorIfOp.thenBlock());
  resultVectorValues.clear();
  for (auto [idx, val] :
       llvm::enumerate(ifOp.thenBlock()->getTerminator()->getOperands())) {
    auto xferWriteOp = val.getDefiningOp<vector::TransferWriteOp>();
    if (!xferWriteOp) {
      return mlir::emitDefiniteFailure(
          this->getOperation(),
          "failed to find xfer write op after vectorization\n");
    }
    resultVectorValues.push_back(xferWriteOp.getVector());
  }
  auto yieldOp = rewriter.create<scf::YieldOp>(loc, resultVectorValues);
  for (auto& op :
       llvm::make_early_inc_range(ifOp.thenBlock()->without_terminator())) {
    op.moveBefore(yieldOp);
    auto xferReadOp = dyn_cast<vector::TransferReadOp>(&op);
    if (!xferReadOp) continue;
    auto it = initTensor2LoadedVectorMap.find(xferReadOp.getSource());
    if (it != initTensor2LoadedVectorMap.end())
      xferReadOp->getResult(0).replaceAllUsesWith(it->second);
  }
  for (auto& e : resultIdx2writeOpMap) {
    target->getResult(e.first).replaceAllUsesWith(e.second->getResult(0));
  }
  ifOp->erase();

  results.push_back({vectorIfOp.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// SplitVectorTransferIntoFullAndPartialOp
//===---------------------------------------------------------------------===//

namespace {

static LogicalResult splitFullAndPartialTransferPrecondition(
    VectorTransferOpInterface xferOp) {
  // TODO: support 0-d corner case.
  if (xferOp.getTransferRank() == 0) return failure();

  // TODO: expand support to these 2 cases.
  if (!xferOp.permutation_map().isMinorIdentity()) return failure();
  // Must have some out-of-bounds dimension to be a candidate for splitting.
  if (!xferOp.hasOutOfBoundsDim()) return failure();
  return success();
}

}  // namespace

DiagnosedSilenceableFailure SplitVectorTransferIntoFullAndPartialOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto xferOp = dyn_cast<VectorTransferOpInterface>(target);
  if (!xferOp) {
    return mlir::emitDefiniteFailure(
        target, "applies only to vector transfer like ops.");
  }

  if (failed(splitFullAndPartialTransferPrecondition(xferOp))) {
    results.push_back({xferOp.getOperation()});
    return DiagnosedSilenceableFailure::success();
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  scf::IfOp ifOp;
  vector::VectorTransformsOptions options;
  options.setVectorTransferSplit(vector::VectorTransferSplit::LinalgCopy);

  if (failed(vector::splitFullAndPartialTransfer(rewriter, xferOp, options,
                                                 &ifOp))) {
    return mlir::emitDefiniteFailure(target, "failed to split xfer ops.");
  }
  results.push_back({ifOp.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// LowerConditionalGenericOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure LowerConditionalGenericOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  auto cgOp = dyn_cast<disc_linalg_ext::ConditionalGenericOp>(target);
  if (!cgOp) {
    return mlir::emitDefiniteFailure(
        target, "applies only to disc_linalg_ext.conditional_generic.");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(target);
  scf::IfOp ifOp =
      rewriter.create<scf::IfOp>(cgOp->getLoc(), cgOp->getOperand(0), false);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  auto genericOp = convertConditionalGenericOpToGenericOp(
      rewriter, dyn_cast<linalg::LinalgOp>(target));
  rewriter.eraseOp(target);
  results.push_back({genericOp});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCLowerVectorsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCLowerVectorsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  MLIRContext* ctx = getContext();
  vector::VectorTransposeLowering vectorTransposeLowering =
      getTransposeLowering();
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      getMultireductionLowering();
  vector::VectorContractLowering vectorContractLowering =
      getContractionLowering();
  vector::VectorTransferSplit vectorTransferSplit = getSplitTransfers();

  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(vectorContractLowering)
      .setVectorMultiReductionLowering(vectorMultiReductionLowering)
      .setVectorTransposeLowering(vectorTransposeLowering)
      .setVectorTransferSplit(vectorTransferSplit);

  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll(getUnrollVectorTransfers());

  int maxTransferRank = 1;

  auto avx2LoweringOptions =
      x86vector::avx2::LoweringOptions().setTransposeOptions(
          x86vector::avx2::TransposeLoweringOptions()
              .lower4x8xf32(getTransposeAvx2Lowering())
              .lower8x8xf32(getTransposeAvx2Lowering()));

  // Note(wyzero): We can not merge two phases into one due to some wierd
  // bugs in some tile configurations. For example, it'll not converge if the
  // tile size of the k is one.

  // phase #1:
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);

    // Stage 1: contraction lowerings.
    patterns.add<mlir::vector::ContractionOpToOuterProductOpLowering,
                 mlir::vector::ContractionOpToMatmulOpLowering,
                 mlir::vector::ContractionOpLowering>(vectorTransformOptions,
                                                      ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

    // Stage 2: multi-reduction lowerings.
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorTransformOptions.vectorMultiReductionLowering);

    // Stage 3: Rewrite vector.transfer into full and partial parts.
    patterns.add<vector::VectorTransferFullPartialRewriter>(
        ctx, vectorTransformOptions);

    // Stage 4: Lower vector transfers.
    vector::populateVectorTransferLoweringPatterns(patterns, maxTransferRank);

    // Apply everything.
    if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
      return DiagnosedSilenceableFailure::definiteFailure();
  }

  // phase #2:
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);

    // Stage 5: Vector to scf patterns.
    populateVectorToSCFConversionPatterns(
        patterns, vectorTransferToSCFOptions.setTargetRank(maxTransferRank));

    // Stage 6: Lower vector.shape_cast.
    vector::populateVectorShapeCastLoweringPatterns(patterns);

    // Stage 7: Lower vector.transpose.
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);

    if (getTransposeAvx2Lowering())
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns, avx2LoweringOptions, /*benefit=*/10);

    // Apply everything.
    if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
      return DiagnosedSilenceableFailure::definiteFailure();
  }

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
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
