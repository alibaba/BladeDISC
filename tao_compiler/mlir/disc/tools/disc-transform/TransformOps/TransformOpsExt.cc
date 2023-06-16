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
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
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
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/disc/tools/disc-transform/ArmNeonExt/ArmNeonExtDialect.h"
#include "mlir/disc/tools/disc-transform/ArmNeonExt/ArmNeonExtOps.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/utils.h"
#include "mlir/disc/transforms/codegen_utils.h"

/// Many code in this file are reused from IREE project, but with customization.

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

static FailureOr<Value> gpuComprehensiveBufferizeAllocationFn(
    OpBuilder& builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpaceAttr);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuComprehensiveBufferizeDeallocationFn(OpBuilder& builder,
                                                             Location loc,
                                                             Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
  return success();
}

static LogicalResult gpuComprehensiveBufferizeCopyFn(OpBuilder& builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // Insert barriers for copies from and to shared memory.
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(from.getType().cast<MemRefType>()) !=
      hasSharedMemoryAddressSpace(to.getType().cast<MemRefType>())) {
    needsBarrier = true;
  }
  if (needsBarrier) {
    builder.create<gpu::BarrierOp>(loc);
  }
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  mlir::disc_ral::createLinalgCopyOp(builder, loc, from, to);
  if (needsBarrier) {
    builder.create<gpu::BarrierOp>(loc);
  }

  return success();
}

OneShotBufferizationOptions getBufferizationOptions(bool target_gpu) {
  OneShotBufferizationOptions options;
  options.allocationFn = target_gpu ? gpuComprehensiveBufferizeAllocationFn
                                    : comprehenciveBufferizeAllocationFn;
  options.deallocationFn = target_gpu ? gpuComprehensiveBufferizeDeallocationFn
                                      : comprehensiveBufferizeDeallocationFn;
  options.memCpyFn = target_gpu ? gpuComprehensiveBufferizeCopyFn
                                : comprehensiveBufferizeCopyFn;
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
  auto options = getBufferizationOptions(getTargetGpu());

  // Bufferize tensor.empty
  if (failed(bufferizeTensorEmptyOps(state, moduleOp, options))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to bufferize tensor.empty.");
  }

  if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options)))
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "bufferization failed.");

  PassManager pm(getContext());
  if (!getTargetGpu()) {
    pm.addNestedPass<func::FuncOp>(
        bufferization::createPromoteBuffersToStackPass([](Value alloc) {
          return betterToUseAlloca(alloc.getType().cast<ShapedType>());
        }));
  }
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

//===---------------------------------------------------------------------===//
// DISCForeachThreadToGPUCTAsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCForeachThreadToGPUCTAsOp::applyToOne(
    mlir::Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  /// from
  /// ```
  /// scf.foreach_thread (%arg3, %arg4) in (%c2, %c2) {
  ///   xxx = some_op(%arg3, %arg4)
  /// } {mapping = [#gpu.block<x>, #gpu.block<y>]}
  /// ```
  /// to
  /// ```
  /// scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c256) step (%c1, %c1) {
  ///   %0:2 = "disc_shape.delinearize"(%arg3, %c2, %c2) : (index, index, index)
  //                                                    -> (index, index)
  ///    xxx = some_op(%0#0, %0#1)
  /// }
  /// ```

  OpBuilder b(target);
  Location loc = target->getLoc();
  MLIRContext* ctx = target->getContext();

  auto foreachThreadOp = dyn_cast<scf::ForeachThreadOp>(target);
  if (!foreachThreadOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to scf::ForeachThreadOp");
  }

  // Only support foreach_thread op with at most rank 3. Otherwise it could not
  // convert to the GPU workgroup.
  int64_t rank = foreachThreadOp.getRank();
  if (rank > 3) {
    return mlir::emitDefiniteFailure(
        target,
        "scf.foreach_thread with rank > 3 does not lower to GPU parallel op");
  }

  // It should contain the mapping information of GPU Block or Warps.
  if (!foreachThreadOp.getMapping().has_value()) {
    return mlir::emitDefiniteFailure(target, "gpu mapping must be present");
  }
  SmallVector<Attribute> gpuMapping =
      llvm::to_vector(foreachThreadOp.getMapping()->getValue());
  if (!llvm::all_of(gpuMapping, [](DeviceMappingAttrInterface map) {
        return map.isa<gpu::GPUBlockMappingAttr>();
      })) {
    return mlir::emitDefiniteFailure(target,
                                     "gpu block mapping must be present");
  }

  Value blockNum = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> ctaDims = foreachThreadOp.getNumThreads();
  for (auto dim : ctaDims) {
    blockNum = b.create<arith::MulIOp>(loc, blockNum, dim);
  }
  // Use block size 128 currently.
  Value threadNum = b.create<arith::ConstantIndexOp>(loc, 128);

  SmallVector<Value, 2> vars;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> lowerBounds{zero, zero};
  SmallVector<Value> upperBounds{blockNum, threadNum};
  SmallVector<Value> steps{one, one};
  auto parallelOp =
      b.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
  // The `linearIdx` will be mapped to CTA index.
  auto linearIdx = parallelOp.getInductionVars()[0];
  b.setInsertionPointToStart(parallelOp.getBody());
  auto mapped_index = calcMultiDimIndex(&b, loc, linearIdx, ctaDims);
  IRMapping mapping;
  for (const auto& z :
       llvm::zip(foreachThreadOp.getThreadIndices(), mapped_index)) {
    mapping.map(std::get<0>(z), std::get<1>(z));
  }
  for (auto& nestedOp : foreachThreadOp.getBody()->without_terminator()) {
    b.clone(nestedOp, mapping);
  }
  if (foreachThreadOp.getResults().size() != parallelOp.getResults().size()) {
    return mlir::emitDefiniteFailure(target,
                                     "the result numbers should be matched");
  }
  for (const auto& z :
       llvm::zip(foreachThreadOp.getResults(), parallelOp.getResults())) {
    std::get<0>(z).replaceAllUsesWith(std::get<1>(z));
  }
  foreachThreadOp.erase();

  parallelOp->setAttr("mapping", StringAttr::get(ctx, "cta-thread-mapping"));

  results.push_back(parallelOp);

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCForeachThreadToGPUWarpsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCForeachThreadToGPUWarpsOp::applyToOne(
    mlir::Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  /// from
  /// ```
  /// scf.foreach_thread (%arg3, %arg4) in (%c2, %c2) {
  ///   xxx = some_op(%arg3, %arg4)
  /// } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  /// ```
  /// to
  /// ```
  /// %warp_linear = gpu.threadX / 32
  /// %warp_idx = disc_shape.delinearize(%warp_linear, %c2, %c2)
  /// xxx = som_op(%warp_idx#0, %warp_idx#1)
  /// ```

  OpBuilder b(target);
  Location loc = target->getLoc();
  MLIRContext* ctx = target->getContext();

  auto foreachThreadOp = dyn_cast<scf::ForeachThreadOp>(target);
  if (!foreachThreadOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to scf::ForeachThreadOp");
  }

  // Only support foreach_thread op with at most rank 3. Otherwise it could not
  // convert to the GPU workgroup.
  int64_t rank = foreachThreadOp.getRank();
  if (rank > 3) {
    return mlir::emitDefiniteFailure(
        target,
        "scf.foreach_thread with rank > 3 does not lower to GPU parallel op");
  }

  // It should contain the mapping information of GPU Block or Warps.
  if (!foreachThreadOp.getMapping().has_value()) {
    return mlir::emitDefiniteFailure(target, "gpu mapping must be present");
  }
  SmallVector<Attribute> gpuMapping =
      llvm::to_vector(foreachThreadOp.getMapping()->getValue());
  if (!llvm::all_of(gpuMapping, [](DeviceMappingAttrInterface map) {
        return map.isa<gpu::GPUThreadMappingAttr>();
      })) {
    // TODO: Use thread mapping to indicate warp mapping currently. To use warp
    // attr after rebase.
    return mlir::emitDefiniteFailure(target,
                                     "gpu warp mapping must be present");
  }

  // It should have a parent parallel op, with attribute `mapping =
  // "cta-thread-mapping"`.
  auto parallelOp = target->getParentOfType<scf::ParallelOp>();
  if (!parallelOp) {
    return mlir::emitDefiniteFailure(
        target,
        "It should have a parent parallel op, with attribute `mapping = "
        "cta-thread-mapping`");
  }
  auto parallelAttr = parallelOp->getAttrOfType<StringAttr>("mapping");
  if (!parallelAttr || !parallelAttr.getValue().equals("cta-thread-mapping")) {
    return mlir::emitDefiniteFailure(
        target, "gpu warp loop must be in CTA parallel op");
  }

  Value threadId = parallelOp.getInductionVars()[1];
  Value warpSize = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value warpId = b.create<arith::DivUIOp>(loc, threadId, warpSize);
  SmallVector<Value> warpDims = foreachThreadOp.getNumThreads();
  auto mappedIndex = calcMultiDimIndex(&b, loc, warpId, warpDims);
  IRMapping mapping;
  for (const auto& z :
       llvm::zip(foreachThreadOp.getThreadIndices(), mappedIndex)) {
    mapping.map(std::get<0>(z), std::get<1>(z));
  }
  for (auto& nestedOp : foreachThreadOp.getBody()->without_terminator()) {
    b.clone(nestedOp, mapping);
  }
  foreachThreadOp.erase();

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCSplitReductionSerialOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCSplitReductionSerialOp::applyToOne(
    mlir::Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  /// Currently, we only support linalg.matmul op.

  auto matmulOp = dyn_cast<linalg::MatmulOp>(target);
  if (!matmulOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to linalg::MatmulOp");
  }

  // TODO: support other types of reduction ops.

  OpBuilder b(target);
  Location loc = target->getLoc();
  MLIRContext* ctx = target->getContext();
  const ArrayRef<int64_t> tileSizes = getTileSizes();
  if (tileSizes.size() != 1) {
    return mlir::emitDefiniteFailure(target,
                                     "only one reduction dim for matmul");
  }
  int64_t staticTileSize = tileSizes[0];

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value step = b.create<arith::ConstantIndexOp>(loc, staticTileSize);
  Value lhs = matmulOp.getDpsInputOperand(0)->get();
  Value rhs = matmulOp.getDpsInputOperand(1)->get();
  Value output = matmulOp.getOutputs()[0];
  Value dimM = b.create<tensor::DimOp>(loc, lhs, zero);
  Value dimN = b.create<tensor::DimOp>(loc, rhs, one);
  Value dimK = b.create<tensor::DimOp>(loc, lhs, one);

  scf::ForOp forOp =
      b.create<scf::ForOp>(loc, zero, dimK, step, ValueRange{output});
  b.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  Value iv = forOp.getInductionVar();
  SmallVector<Value> lhsOffsets{zero, iv};
  SmallVector<Value> lhsDimUppers{dimM, step};
  SmallVector<Value> lhsStrides{one, one};
  Value lhsSlice = b.create<tensor::ExtractSliceOp>(loc, lhs, lhsOffsets,
                                                    lhsDimUppers, lhsStrides);
  SmallVector<Value> rhsOffsets{iv, zero};
  SmallVector<Value> rhsDimUppers{step, dimN};
  SmallVector<Value> rhsStrides{one, one};
  Value rhsSlice = b.create<tensor::ExtractSliceOp>(loc, rhs, rhsOffsets,
                                                    rhsDimUppers, rhsStrides);
  ShapedType resultType = output.getType().cast<ShapedType>();
  Value iterArg = forOp.getRegionIterArg(0);
  linalg::MatmulOp res = b.create<linalg::MatmulOp>(
      loc, resultType, ValueRange{lhsSlice, rhsSlice}, ValueRange{iterArg});
  b.create<scf::YieldOp>(loc, ValueRange{res.getResult(0)});

  target->getResult(0).replaceAllUsesWith(forOp->getResult(0));
  results.push_back(res);
  results.push_back(forOp);

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCVectorToMMAConversionOp
//===---------------------------------------------------------------------===//

void transform_dialect::DISCVectorToMMAConversionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::DISCVectorToMMAConversionOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  // TODO: use createConvertVectorToGPUPass pass.

  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it "
        "needs to apply "
        "patterns greedily");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext* ctx = target->getContext();

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  mlir::populatePrepareVectorToMMAPatterns(patterns, true);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(target))) {
    target->emitOpError("vector to mma patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }
  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(target->getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                          nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(f32ToTF32patterns)))) {
    target->emitOpError("vector to mma F32ToTF32 patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DISCPromoteDotOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::DISCPromoteDotOperandsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  Location loc = target->getLoc();
  IRRewriter rewriter(getContext());
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(target);
  SmallVector<int64_t> indices = llvm::to_vector(getIndices());
  int64_t numOperands = target->getNumOperands();

  results.push_back(target);
  bufferization::BufferizationOptions options;
  for (int64_t index : indices) {
    if ((index >= 0) && (index < numOperands)) {
      FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
          rewriter, loc, target->getOperand(index), false, options, true);
      if (failed(ret)) {
        return emitDefaultDefiniteFailure(target)
               << "failed to promote operand";
      }
      target->setOperand(index, ret.value());
      results.push_back(ret.value().getDefiningOp());
    } else {
      return emitDefaultDefiniteFailure(target) << "invalid index specified";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DISCLowerGmemToSmemOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::DISCLowerGmemToSmemOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  SimplePatternRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  Location loc = target->getLoc();
  MLIRContext* ctx = target->getContext();

  auto genericOp = dyn_cast<linalg::GenericOp>(target);
  if (!genericOp) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to linalg::GenericOp");
  }

  // TODO: check it is gmem to smem movement.

  auto forOp = linalgOpToLoops(rewriter, genericOp);
  if (failed(forOp)) {
    return mlir::emitDefiniteFailure(target, "convert to forOp failed");
  }
  // TODO: convert forOp to instructions mapped to threads.
  target->erase();

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DISCEraseDeallocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::DISCEraseDeallocOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  // This op can be used to erase dealloc ops after bufferization on GPU.

  auto funcOp = cast<func::FuncOp>(target);

  SmallVector<memref::DeallocOp> deallocOps;
  funcOp.walk([&](memref::DeallocOp op) { deallocOps.push_back(op); });

  for (auto op : deallocOps) {
    op->erase();
  }

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DISCTransferWriteZeroToSCFOp
//===----------------------------------------------------------------------===//

struct TransferWriteZeroToSCFPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter& rewriter) const override {
    // To check whether it reads constant vector.
    auto constant = op.getVector().getDefiningOp<arith::ConstantOp>();
    if (!constant) {
      return failure();
    }

    // To check whether it reads zeros.
    auto value = constant.getValue();
    auto denseAttr = value.cast<DenseElementsAttr>();
    bool isSplat = denseAttr.isSplat();
    auto type = constant->getResultTypes()[0];
    auto shapedType = type.dyn_cast<ShapedType>();
    if (!isSplat && (shapedType && shapedType.getRank() != 0)) {
      return failure();
    }
    auto elemTy = shapedType ? shapedType.getElementType() : type;
    bool isZero = false;
    if (elemTy.isIntOrIndex()) {
      isZero = isSplat ? denseAttr.getSplatValue<APInt>().isZero()
                       : denseAttr.getValues<APInt>()[{}].isZero();
    } else if (isa<mlir::FloatType>(elemTy)) {
      isZero = isSplat ? denseAttr.getSplatValue<APFloat>().isZero()
                       : denseAttr.getValues<APFloat>()[{}].isZero();
    }
    if (!isZero) {
      return failure();
    }

    // Currently, it only supports memref type.
    auto source = op.getSource();
    auto sourceType = source.getType().dyn_cast<MemRefType>();
    if (!sourceType) {
      return failure();
    }
    // To guarantee that it is an identity write on memref.
    // Do not support permutation currently. Do not support mask.
    if (!op.getPermutationMap().isIdentity() || op.getMask()) {
      return failure();
    }
    // TODO: check in_bounds.

    // The op should be not used by other ops, as we will erase it directly
    // after creating the loop.
    if (!op->getUses().empty()) {
      return failure();
    }

    // Create scf.loop and memref.store op.
    int rank = sourceType.getRank();
    if (rank < 1) {
      return failure();
    }
    auto loc = op.getLoc();
    Value zeroValue = elemTy.isIntOrIndex()
                          ? rewriter.create<arith::ConstantOp>(
                                loc, rewriter.getIntegerAttr(elemTy, 0))
                          : rewriter.create<arith::ConstantOp>(
                                loc, rewriter.getFloatAttr(elemTy, 0));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> indices(rank);
    for (int i = 0; i < rank; i++) {
      auto dim = rewriter.create<memref::DimOp>(loc, source, i);
      auto forOp =
          rewriter.create<scf::ForOp>(loc, zero, dim, one, ValueRange{});
      indices[i] = forOp.getInductionVar();
      Block& newBlock = forOp.getRegion().front();
      rewriter.setInsertionPointToStart(&newBlock);
    }
    rewriter.create<memref::StoreOp>(loc, zeroValue, source, indices);

    // Erase the op.
    op->erase();

    return success();
  }
};  // namespace transform_dialect

DiagnosedSilenceableFailure
transform_dialect::DISCTransferWriteZeroToSCFOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  MLIRContext* ctx = getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<TransferWriteZeroToSCFPattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform_dialect::DISCInlineAndConvertGPUIdsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  scf::ParallelOp ctaParallelOp;
  target->walk([&](scf::ParallelOp op) {
    auto parallelAttr = op->getAttrOfType<StringAttr>("mapping");
    if (parallelAttr && parallelAttr.getValue().equals("cta-thread-mapping")) {
      ctaParallelOp = op;
      WalkResult::interrupt();
    }
  });
  if (!ctaParallelOp) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  SmallVector<Operation*> candidateOps;
  target->walk([&](Operation* op) {
    if (op->getBlock() == ctaParallelOp->getBlock() &&
        op->isBeforeInBlock(ctaParallelOp)) {
      candidateOps.push_back(op);
    }
  });
  OpBuilder b(target);
  auto loc = target->getLoc();
  for (auto op : candidateOps) {
    // TODO: inline and convert more ops.
    if (isa<gpu::LaneIdOp>(op)) {
      b.setInsertionPoint(&ctaParallelOp.getBody()->front());
      Value threadId = ctaParallelOp.getInductionVars()[1];
      Value warpSize = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
      Value laneId = b.create<arith::RemUIOp>(loc, threadId, warpSize);
      op->getResult(0).replaceAllUsesWith(laneId);
    }
  }

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCLowerVectorContractionToBFMMLA2x2x4
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCLowerVectorContractionToBFMMLA2x2x4::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  vector::ContractionOp vec_contract = dyn_cast<vector::ContractionOp>(target);
  if (!vec_contract) {
    return mlir::emitDefiniteFailure(target,
                                     "apples only to vector.contract op.");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  Location loc = vec_contract.getLoc();
  rewriter.setInsertionPoint(target);
  auto extf1 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(0).getDefiningOp());
  auto extf2 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(1).getDefiningOp());
  if (!extf1 || !extf2) {
    return mlir::emitDefiniteFailure(target, "require extf bf16 to f32.");
  }
  int64_t m =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[0];
  int64_t k =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[1];
  int64_t n =
      vec_contract.getOperand(1).getType().cast<VectorType>().getShape()[0];
  if (m != 2 || n != 2 || k != 4) {
    return mlir::emitDefiniteFailure(target, "m, n, k not satisfied.");
  }

  // flatten 2x4xbf16 to 8xbf16
  VectorType flattenedVectorType =
      VectorType::get({m * k}, FloatType::getBF16(ctx));
  flattenedVectorType.dump();
  Value a1d = rewriter.create<vector::ShapeCastOp>(loc, flattenedVectorType,
                                                   extf1->getOperand(0));
  Value b1d = rewriter.create<vector::ShapeCastOp>(loc, flattenedVectorType,
                                                   extf2->getOperand(0));
  // flatten 2x2xf32 to 4xf32
  Value c1d = rewriter.create<vector::ShapeCastOp>(
      loc, VectorType::get({m * n}, FloatType::getF32(ctx)),
      vec_contract.getOperand(2));

  auto bfmmla_op = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      vec_contract.getLoc(), VectorType::get({m * n}, FloatType::getF32(ctx)),
      c1d, a1d, b1d);
  vec_contract.getResult().getType().cast<VectorType>().dump();
  auto res = rewriter.create<vector::ShapeCastOp>(
      loc, vec_contract.getResult().getType().cast<VectorType>(), bfmmla_op);
  target->replaceAllUsesWith(res);
  results.push_back({res.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCLowerVectorContractionToBFMMLA4x4x8
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCLowerVectorContractionToBFMMLA4x4x8::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  vector::ContractionOp vec_contract = dyn_cast<vector::ContractionOp>(target);
  if (!vec_contract) {
    return mlir::emitDefiniteFailure(target,
                                     "apples only to vector.contract op.");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  Location loc = vec_contract.getLoc();
  rewriter.setInsertionPoint(target);
  auto extf1 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(0).getDefiningOp());
  auto extf2 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(1).getDefiningOp());
  if (!extf1 || !extf2) {
    return mlir::emitDefiniteFailure(target, "require extf bf16 to f32.");
  }
  int64_t m =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[0];
  int64_t k =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[1];
  int64_t n =
      vec_contract.getOperand(1).getType().cast<VectorType>().getShape()[0];
  if (m != 4 || n != 4 || k != 8) {
    return mlir::emitDefiniteFailure(target, "m, n, k not satisfied.");
  }
  using I64Array = ArrayRef<int64_t>;
  SmallVector<int64_t, 2> offsets_0 = {0, 0};
  SmallVector<int64_t, 2> offsets_1 = {1, 0};
  SmallVector<int64_t, 2> offsets_2 = {2, 0};
  SmallVector<int64_t, 2> offsets_3 = {3, 0};
  SmallVector<int64_t, 2> sizes = {1, 8};
  SmallVector<int64_t, 2> strides = {1, 1};

  auto f64x2_vecTy = VectorType::get({2}, FloatType::getF64(ctx));    // 2xf64
  auto f32x4_vecTy = VectorType::get({4}, FloatType::getF32(ctx));    // 4xf32
  auto bf16x8_vecTy = VectorType::get({8}, FloatType::getBF16(ctx));  // 8xbf16

  // Preprocessing A
  // extract 8xbf16 from 4x8xbf16
  auto ASliceOp0 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{0, 0}, sizes, strides));
  auto ASliceOp1 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{1, 0}, sizes, strides));
  auto ASliceOp2 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{2, 0}, sizes, strides));
  auto ASliceOp3 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{3, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp0);
  auto ASliceOp1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp1);
  auto ASliceOp2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp2);
  auto ASliceOp3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp3);
  // UZP1 and UZP2
  auto ASlicpOP0UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp0Bitcast, ASliceOp1Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASlicpOP1UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp0Bitcast, ASliceOp1Bitcast,
      BoolAttr::get(ctx, 0));
  auto ASlicpOP2UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp2Bitcast, ASliceOp3Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASlicpOP3UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp2Bitcast, ASliceOp3Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A0 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP0UZP1);
  auto A1 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP1UZP2);
  auto A2 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP2UZP1);
  auto A3 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP3UZP2);

  // Preprocessing B
  // extract 8xbf16 from 4x8xbf16
  auto BSliceOp0 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), offsets_0, sizes, strides));
  auto BSliceOp1 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), offsets_1, sizes, strides));
  auto BSliceOp2 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), offsets_2, sizes, strides));
  auto BSliceOp3 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), offsets_3, sizes, strides));
  // 8xbf16 -> 2xf64
  auto BSliceOp0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp0);
  auto BSliceOp1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp1);
  auto BSliceOp2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp2);
  auto BSliceOp3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp3);
  // UZP1 and UZP2
  auto BSliceOp0UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp0Bitcast, BSliceOp1Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp1UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp0Bitcast, BSliceOp1Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp2UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp2Bitcast, BSliceOp3Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp3UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp2Bitcast, BSliceOp3Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto B0 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp0UZP1);
  auto B1 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp1UZP2);
  auto B2 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp2UZP1);
  auto B3 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp3UZP2);

  // Preprocessing C
  Value acc = vec_contract.getOperand(2);  // 4x4xf32
  SmallVector<int64_t, 2> C_sizes = {1, 4};
  // extract 4xf32 from 4x4xf32
  auto CSliceOp0 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, offsets_0,
                                                     C_sizes, strides));
  auto CSliceOp1 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, offsets_1,
                                                     C_sizes, strides));
  auto CSliceOp2 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, offsets_2,
                                                     C_sizes, strides));
  auto CSliceOp3 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, offsets_3,
                                                     C_sizes, strides));
  // 4xf32 -> 2xf64
  auto CSliceOp0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp0);
  auto CSliceOp1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp1);
  auto CSliceOp2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp2);
  auto CSliceOp3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp3);
  // UZP1 and UZP2
  auto C0UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp0Bitcast, CSliceOp1Bitcast,
      BoolAttr::get(ctx, 1));
  auto C1UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp0Bitcast, CSliceOp1Bitcast,
      BoolAttr::get(ctx, 0));
  auto C2UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp2Bitcast, CSliceOp3Bitcast,
      BoolAttr::get(ctx, 1));
  auto C3UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp2Bitcast, CSliceOp3Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  auto C0 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C0UZP);
  auto C1 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C1UZP);
  auto C2 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C2UZP);
  auto C3 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C3UZP);

  // BFMMLA
  auto bfmmla_op_0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, C0, A0, B0);
  auto bfmmla_op_1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, C1, A0, B2);
  auto bfmmla_op_2 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, C2, A2, B0);
  auto bfmmla_op_3 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, C3, A2, B2);
  bfmmla_op_0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, bfmmla_op_0, A1, B1);
  bfmmla_op_1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, bfmmla_op_1, A1, B3);
  bfmmla_op_2 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, bfmmla_op_2, A3, B1);
  bfmmla_op_3 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(
      loc, f32x4_vecTy, bfmmla_op_3, A3, B3);

  // Postprocessing C
  // 4xf32 -> 2xf64
  auto C0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, bfmmla_op_0);
  auto C1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, bfmmla_op_1);
  auto C2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, bfmmla_op_2);
  auto C3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, bfmmla_op_3);
  // UZP1 and UZP2
  C0UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, C0Bitcast, C1Bitcast, BoolAttr::get(ctx, 1));
  C1UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, C0Bitcast, C1Bitcast, BoolAttr::get(ctx, 0));
  C2UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, C2Bitcast, C3Bitcast, BoolAttr::get(ctx, 1));
  C3UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, C2Bitcast, C3Bitcast, BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C0Bitcast = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C0UZP);
  C1Bitcast = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C1UZP);
  C2Bitcast = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C2UZP);
  C3Bitcast = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C3UZP);

  SmallVector<int64_t, 2> C_strides = {1};
  auto res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C0Bitcast, acc, offsets_0, ArrayRef<int64_t>{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(loc, C1Bitcast, res,
                                                      offsets_1, C_strides);
  res = rewriter.create<vector::InsertStridedSliceOp>(loc, C2Bitcast, res,
                                                      offsets_2, C_strides);
  res = rewriter.create<vector::InsertStridedSliceOp>(loc, C3Bitcast, res,
                                                      offsets_3, C_strides);

  target->replaceAllUsesWith(res);
  results.push_back(res.getOperation());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCLowerVectorContractionToBFMMLA8x8x8
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure DISCLowerVectorContractionToBFMMLA8x8x8::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  vector::ContractionOp vec_contract = dyn_cast<vector::ContractionOp>(target);
  if (!vec_contract) {
    return mlir::emitDefiniteFailure(target,
                                     "apples only to vector.contract op.");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  Location loc = vec_contract.getLoc();
  rewriter.setInsertionPoint(target);
  auto extf1 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(0).getDefiningOp());
  auto extf2 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(1).getDefiningOp());
  if (!extf1 || !extf2) {
    return mlir::emitDefiniteFailure(target, "require extf bf16 to f32.");
  }
  int64_t m =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[0];
  int64_t k =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[1];
  int64_t n =
      vec_contract.getOperand(1).getType().cast<VectorType>().getShape()[0];
  if (m != 8 || n != 8 || k != 8) {
    return mlir::emitDefiniteFailure(target, "m, n, k not satisfied.");
  }
  using I64Array = ArrayRef<int64_t>;
  SmallVector<int64_t, 2> sizes = {1, 8};
  SmallVector<int64_t, 2> strides = {1, 1};

  auto f64x2_vecTy = VectorType::get({2}, FloatType::getF64(ctx));    // 2xf64
  auto f32x4_vecTy = VectorType::get({4}, FloatType::getF32(ctx));    // 4xf32
  auto bf16x8_vecTy = VectorType::get({8}, FloatType::getBF16(ctx));  // 8xbf16

  // Preprocessing B
  // extract 8xbf16 from 8x8xbf16
  auto BSliceOp0 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{0, 0}, sizes, strides));
  auto BSliceOp1 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{1, 0}, sizes, strides));
  auto BSliceOp2 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{2, 0}, sizes, strides));
  auto BSliceOp3 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{3, 0}, sizes, strides));
  auto BSliceOp4 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{4, 0}, sizes, strides));
  auto BSliceOp5 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{5, 0}, sizes, strides));
  auto BSliceOp6 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{6, 0}, sizes, strides));
  auto BSliceOp7 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf2.getOperand(), I64Array{7, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto BSliceOp0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp0);
  auto BSliceOp1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp1);
  auto BSliceOp2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp2);
  auto BSliceOp3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp3);
  auto BSliceOp4Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp4);
  auto BSliceOp5Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp5);
  auto BSliceOp6Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp6);
  auto BSliceOp7Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp7);
  // UZP1 and UZP2
  auto BSliceOp0UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp0Bitcast, BSliceOp1Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp1UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp0Bitcast, BSliceOp1Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp2UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp2Bitcast, BSliceOp3Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp3UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp2Bitcast, BSliceOp3Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp4UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp4Bitcast, BSliceOp5Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp5UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp4Bitcast, BSliceOp5Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp6UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp6Bitcast, BSliceOp7Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp7UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp6Bitcast, BSliceOp7Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto B0 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp0UZP1);
  auto B1 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp1UZP2);
  auto B2 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp2UZP1);
  auto B3 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp3UZP2);
  auto B4 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp4UZP1);
  auto B5 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp5UZP2);
  auto B6 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp6UZP1);
  auto B7 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp7UZP2);

  // Preprocessing C
  Value acc = vec_contract.getOperand(2);  // 8x8xf32
  SmallVector<int64_t, 2> C_sizes = {1, 4};
  // extract 16 4xf32 from 8x8xf32
  auto CSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{0, 0},
                                                     C_sizes, strides));
  auto CSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{0, 4},
                                                     C_sizes, strides));
  auto CSliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{1, 0},
                                                     C_sizes, strides));
  auto CSliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{1, 4},
                                                     C_sizes, strides));
  auto CSliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{2, 0},
                                                     C_sizes, strides));
  auto CSliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{2, 4},
                                                     C_sizes, strides));
  auto CSliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{3, 0},
                                                     C_sizes, strides));
  auto CSliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{3, 4},
                                                     C_sizes, strides));
  auto CSliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{4, 0},
                                                     C_sizes, strides));
  auto CSliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{4, 4},
                                                     C_sizes, strides));
  auto CSliceOp10 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{5, 0},
                                                     C_sizes, strides));
  auto CSliceOp11 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{5, 4},
                                                     C_sizes, strides));
  auto CSliceOp12 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{6, 0},
                                                     C_sizes, strides));
  auto CSliceOp13 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{6, 4},
                                                     C_sizes, strides));
  auto CSliceOp14 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{7, 0},
                                                     C_sizes, strides));
  auto CSliceOp15 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{7, 4},
                                                     C_sizes, strides));
  // 4xf32 -> 2xf64
  auto CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp00);
  auto CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp01);
  auto CSliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp02);
  auto CSliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp03);
  auto CSliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp04);
  auto CSliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp05);
  auto CSliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp06);
  auto CSliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp07);
  auto CSliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp08);
  auto CSliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp09);
  auto CSliceOp10Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp10);
  auto CSliceOp11Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp11);
  auto CSliceOp12Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp12);
  auto CSliceOp13Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp13);
  auto CSliceOp14Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp14);
  auto CSliceOp15Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp15);
  // UZP1 and UZP2
  auto C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp02Bitcast,
      BoolAttr::get(ctx, 1));
  auto C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp02Bitcast,
      BoolAttr::get(ctx, 0));
  auto C02UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp01Bitcast, CSliceOp03Bitcast,
      BoolAttr::get(ctx, 1));
  auto C03UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp01Bitcast, CSliceOp03Bitcast,
      BoolAttr::get(ctx, 0));
  auto C04UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp04Bitcast, CSliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  auto C05UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp04Bitcast, CSliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  auto C06UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp05Bitcast, CSliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  auto C07UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp05Bitcast, CSliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  auto C08UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp08Bitcast, CSliceOp10Bitcast,
      BoolAttr::get(ctx, 1));
  auto C09UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp08Bitcast, CSliceOp10Bitcast,
      BoolAttr::get(ctx, 0));
  auto C10UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp09Bitcast, CSliceOp11Bitcast,
      BoolAttr::get(ctx, 1));
  auto C11UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp09Bitcast, CSliceOp11Bitcast,
      BoolAttr::get(ctx, 0));
  auto C12UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp12Bitcast, CSliceOp14Bitcast,
      BoolAttr::get(ctx, 1));
  auto C13UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp12Bitcast, CSliceOp14Bitcast,
      BoolAttr::get(ctx, 0));
  auto C14UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp13Bitcast, CSliceOp15Bitcast,
      BoolAttr::get(ctx, 1));
  auto C15UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp13Bitcast, CSliceOp15Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  auto C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  auto C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  auto C02 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C02UZP);
  auto C03 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C03UZP);
  auto C04 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C04UZP);
  auto C05 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C05UZP);
  auto C06 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C06UZP);
  auto C07 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C07UZP);
  auto C08 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C08UZP);
  auto C09 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C09UZP);
  auto C10 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C10UZP);
  auto C11 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C11UZP);
  auto C12 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C12UZP);
  auto C13 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C13UZP);
  auto C14 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C14UZP);
  auto C15 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C15UZP);

  // Preprocessing A
  // extract 8xbf16 from 4x8xbf16
  auto ASliceOp0 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{0, 0}, sizes, strides));
  auto ASliceOp1 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{1, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp0Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp0);
  auto ASliceOp1Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp1);
  // UZP1 and UZP2
  auto ASlicpOP0UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp0Bitcast, ASliceOp1Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASlicpOP1UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp0Bitcast, ASliceOp1Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A0 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP0UZP1);
  auto A1 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP1UZP2);

  // Micro Kernel Computation
  auto BFMMLA00 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C00, A0, B0);
  BFMMLA00 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA00, A1, B1);
  auto BFMMLA01 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C01, A0, B2);
  BFMMLA01 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA01, A1, B3);
  auto BFMMLA02 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C02, A0, B4);
  BFMMLA02 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA02, A1, B5);
  auto BFMMLA03 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C03, A0, B6);
  BFMMLA03 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA03, A1, B7);

  // Preprocessing A
  // extract 8xbf16 from 4x8xbf16
  auto ASliceOp2 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{2, 0}, sizes, strides));
  auto ASliceOp3 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{3, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp2Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp2);
  auto ASliceOp3Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp3);
  // UZP1 and UZP2
  auto ASlicpOP2UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp2Bitcast, ASliceOp3Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASlicpOP3UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp2Bitcast, ASliceOp3Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A2 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP2UZP1);
  auto A3 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASlicpOP3UZP2);

  // Micro Kernel Computation
  auto BFMMLA04 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C04, A2, B0);
  BFMMLA04 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA04, A3, B1);
  auto BFMMLA05 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C05, A2, B2);
  BFMMLA05 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA05, A3, B3);
  auto BFMMLA06 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C06, A2, B4);
  BFMMLA06 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA06, A3, B5);
  auto BFMMLA07 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C07, A2, B6);
  BFMMLA07 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA07, A3, B7);

  // Preprocessing A
  // extract 8xbf16 from 4x8xbf16
  auto ASliceOp4 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{4, 0}, sizes, strides));
  auto ASliceOp5 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{5, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp4Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp4);
  auto ASliceOp5Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp5);
  // UZP1 and UZP2
  auto ASliceOp4UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp4Bitcast, ASliceOp5Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp5UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp4Bitcast, ASliceOp5Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A4 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp4UZP1);
  auto A5 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp5UZP2);

  // Micro Kernel Computation
  auto BFMMLA08 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C08, A4, B0);
  BFMMLA08 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA08, A5, B1);
  auto BFMMLA09 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C09, A4, B2);
  BFMMLA09 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA09, A5, B3);
  auto BFMMLA10 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C10, A4, B4);
  BFMMLA10 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA10, A5, B5);
  auto BFMMLA11 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C11, A4, B6);
  BFMMLA11 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA11, A5, B7);

  // Preprocessing A
  // extract 8xbf16 from 4x8xbf16
  auto ASliceOp6 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{6, 0}, sizes, strides));
  auto ASliceOp7 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(
          loc, extf1.getOperand(), I64Array{7, 0}, sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp6Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp6);
  auto ASliceOp7Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp7);
  // UZP1 and UZP2
  auto ASliceOp6UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp6Bitcast, ASliceOp7Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp7UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp6Bitcast, ASliceOp7Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A6 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp6UZP1);
  auto A7 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp7UZP2);

  // Micro Kernel Computation
  auto BFMMLA12 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C12, A6, B0);
  BFMMLA12 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA12, A7, B1);
  auto BFMMLA13 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C13, A6, B2);
  BFMMLA13 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA13, A7, B3);
  auto BFMMLA14 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C14, A6, B4);
  BFMMLA14 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA14, A7, B5);
  auto BFMMLA15 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                               C15, A6, B6);
  BFMMLA15 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                          BFMMLA15, A7, B7);

  // Postprocessing C
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA00);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA01);
  CSliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA02);
  CSliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA03);
  CSliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA04);
  CSliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA05);
  CSliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA06);
  CSliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA07);
  CSliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA08);
  CSliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA09);
  CSliceOp10Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA10);
  CSliceOp11Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA11);
  CSliceOp12Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA12);
  CSliceOp13Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA13);
  CSliceOp14Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA14);
  CSliceOp15Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA15);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  C02UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp02Bitcast, CSliceOp03Bitcast,
      BoolAttr::get(ctx, 1));
  C03UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp02Bitcast, CSliceOp03Bitcast,
      BoolAttr::get(ctx, 0));
  C04UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp04Bitcast, CSliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  C05UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp04Bitcast, CSliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  C06UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp06Bitcast, CSliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  C07UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp06Bitcast, CSliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  C08UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp08Bitcast, CSliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  C09UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp08Bitcast, CSliceOp09Bitcast,
      BoolAttr::get(ctx, 0));
  C10UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp10Bitcast, CSliceOp11Bitcast,
      BoolAttr::get(ctx, 1));
  C11UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp10Bitcast, CSliceOp11Bitcast,
      BoolAttr::get(ctx, 0));
  C12UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp12Bitcast, CSliceOp13Bitcast,
      BoolAttr::get(ctx, 1));
  C13UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp12Bitcast, CSliceOp13Bitcast,
      BoolAttr::get(ctx, 0));
  C14UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp14Bitcast, CSliceOp15Bitcast,
      BoolAttr::get(ctx, 1));
  C15UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp14Bitcast, CSliceOp15Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  C02 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C02UZP);
  C03 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C03UZP);
  C04 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C04UZP);
  C05 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C05UZP);
  C06 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C06UZP);
  C07 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C07UZP);
  C08 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C08UZP);
  C09 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C09UZP);
  C10 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C10UZP);
  C11 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C11UZP);
  C12 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C12UZP);
  C13 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C13UZP);
  C14 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C14UZP);
  C15 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C15UZP);
  // Insert 4xf32 into 8x8xf32
  vector::InsertStridedSliceOp res;
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C00, acc, I64Array{0, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C02, res, I64Array{0, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C01, res, I64Array{1, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C03, res, I64Array{1, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C04, res, I64Array{2, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C06, res, I64Array{2, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C05, res, I64Array{3, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C07, res, I64Array{3, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C08, res, I64Array{4, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C10, res, I64Array{4, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C09, res, I64Array{5, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C11, res, I64Array{5, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C12, res, I64Array{6, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C14, res, I64Array{6, 4}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C13, res, I64Array{7, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C15, res, I64Array{7, 4}, I64Array{1});

  target->replaceAllUsesWith(res);
  results.push_back(res.getOperation());
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DISCLowerVectorContractionToBFMMLA8x4x40
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
DISCLowerVectorContractionToBFMMLA8x4x40::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  vector::ContractionOp vec_contract = dyn_cast<vector::ContractionOp>(target);
  if (!vec_contract) {
    return mlir::emitDefiniteFailure(target,
                                     "apples only to vector.contract op.");
  }

  MLIRContext* ctx = target->getContext();
  IRRewriter rewriter(ctx);
  Location loc = vec_contract.getLoc();
  rewriter.setInsertionPoint(target);
  auto extf1 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(0).getDefiningOp());
  auto extf2 =
      dyn_cast<arith::ExtFOp>(vec_contract.getOperand(1).getDefiningOp());
  if (!extf1 || !extf2) {
    return mlir::emitDefiniteFailure(target, "require extf bf16 to f32.");
  }
  int64_t m =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[0];
  int64_t k =
      vec_contract.getOperand(0).getType().cast<VectorType>().getShape()[1];
  int64_t n =
      vec_contract.getOperand(1).getType().cast<VectorType>().getShape()[0];
  if (m != 8 || n != 4 || k != 40) {
    return mlir::emitDefiniteFailure(target, "m, n, k not satisfied.");
  }

  using I64Array = ArrayRef<int64_t>;
  SmallVector<int64_t, 2> sizes = {1, 8};
  SmallVector<int64_t, 2> strides = {1, 1};
  auto f64x2_vecTy = VectorType::get({2}, FloatType::getF64(ctx));    // 2xf64
  auto f32x4_vecTy = VectorType::get({4}, FloatType::getF32(ctx));    // 4xf32
  auto bf16x8_vecTy = VectorType::get({8}, FloatType::getBF16(ctx));  // 8xbf16

  Value A = extf1.getOperand(), B = extf2.getOperand();
  Value acc = vec_contract.getOperand(2);
  vector::InsertStridedSliceOp res;

  // Preprocessing B
  // extract 8xbf16 from 4x40xbf16
  auto BSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{0, 0},
                                                     sizes, strides));
  auto BSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{0, 8},
                                                     sizes, strides));
  auto BSliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{0, 16},
                                                     sizes, strides));
  auto BSliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{0, 24},
                                                     sizes, strides));
  auto BSliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{0, 32},
                                                     sizes, strides));
  auto BSliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{1, 0},
                                                     sizes, strides));
  auto BSliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{1, 8},
                                                     sizes, strides));
  auto BSliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{1, 16},
                                                     sizes, strides));
  auto BSliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{1, 24},
                                                     sizes, strides));
  auto BSliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{1, 32},
                                                     sizes, strides));

  auto BSliceOp10 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{2, 0},
                                                     sizes, strides));
  auto BSliceOp11 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{2, 8},
                                                     sizes, strides));
  auto BSliceOp12 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{2, 16},
                                                     sizes, strides));
  auto BSliceOp13 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{2, 24},
                                                     sizes, strides));
  auto BSliceOp14 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{2, 32},
                                                     sizes, strides));
  auto BSliceOp15 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{3, 0},
                                                     sizes, strides));
  auto BSliceOp16 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{3, 8},
                                                     sizes, strides));
  auto BSliceOp17 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{3, 16},
                                                     sizes, strides));
  auto BSliceOp18 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{3, 24},
                                                     sizes, strides));
  auto BSliceOp19 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, B, I64Array{3, 32},
                                                     sizes, strides));
  // 8xbf16 -> 2xf64
  auto BSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp00);
  auto BSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp01);
  auto BSliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp02);
  auto BSliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp03);
  auto BSliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp04);
  auto BSliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp05);
  auto BSliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp06);
  auto BSliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp07);
  auto BSliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp08);
  auto BSliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp09);

  auto BSliceOp10Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp10);
  auto BSliceOp11Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp11);
  auto BSliceOp12Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp12);
  auto BSliceOp13Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp13);
  auto BSliceOp14Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp14);
  auto BSliceOp15Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp15);
  auto BSliceOp16Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp16);
  auto BSliceOp17Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp17);
  auto BSliceOp18Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp18);
  auto BSliceOp19Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BSliceOp19);
  // UZP1 and UZP2
  auto BSliceOp00UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp00Bitcast, BSliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp01UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp00Bitcast, BSliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp02UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp01Bitcast, BSliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp03UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp01Bitcast, BSliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp04UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp02Bitcast, BSliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp05UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp02Bitcast, BSliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp06UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp03Bitcast, BSliceOp08Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp07UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp03Bitcast, BSliceOp08Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp08UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp04Bitcast, BSliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp09UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp04Bitcast, BSliceOp09Bitcast,
      BoolAttr::get(ctx, 0));

  auto BSliceOp10UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp10Bitcast, BSliceOp15Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp11UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp10Bitcast, BSliceOp15Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp12UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp11Bitcast, BSliceOp16Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp13UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp11Bitcast, BSliceOp16Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp14UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp12Bitcast, BSliceOp17Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp15UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp12Bitcast, BSliceOp17Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp16UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp13Bitcast, BSliceOp18Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp17UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp13Bitcast, BSliceOp18Bitcast,
      BoolAttr::get(ctx, 0));
  auto BSliceOp18UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp14Bitcast, BSliceOp19Bitcast,
      BoolAttr::get(ctx, 1));
  auto BSliceOp19UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, BSliceOp14Bitcast, BSliceOp19Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto B00 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp00UZP1);
  auto B01 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp01UZP2);
  auto B02 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp02UZP1);
  auto B03 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp03UZP2);
  auto B04 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp04UZP1);
  auto B05 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp05UZP2);
  auto B06 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp06UZP1);
  auto B07 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp07UZP2);
  auto B08 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp08UZP1);
  auto B09 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp09UZP2);

  auto B10 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp10UZP1);
  auto B11 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp11UZP2);
  auto B12 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp12UZP1);
  auto B13 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp13UZP2);
  auto B14 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp14UZP1);
  auto B15 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp15UZP2);
  auto B16 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp16UZP1);
  auto B17 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp17UZP2);
  auto B18 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp18UZP1);
  auto B19 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, BSliceOp19UZP2);

  // Preprocessing C
  // extract 4xf32 from 2x4xf32
  auto CSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{0, 0},
                                                     I64Array{1, 4}, strides));
  auto CSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{1, 0},
                                                     I64Array{1, 4}, strides));
  // 4xf32 -> 2xf64
  auto CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp00);
  auto CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp01);
  // UZP1 and UZP2
  auto C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  auto C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  auto C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  auto C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);

  // Preprocessing A
  // extract 8xbf16 from 2x40xbf16
  auto ASliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{0, 0},
                                                     sizes, strides));
  auto ASliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{0, 8},
                                                     sizes, strides));
  auto ASliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{0, 16},
                                                     sizes, strides));
  auto ASliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{0, 24},
                                                     sizes, strides));
  auto ASliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{0, 32},
                                                     sizes, strides));
  auto ASliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{1, 0},
                                                     sizes, strides));
  auto ASliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{1, 8},
                                                     sizes, strides));
  auto ASliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{1, 16},
                                                     sizes, strides));
  auto ASliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{1, 24},
                                                     sizes, strides));
  auto ASliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{1, 32},
                                                     sizes, strides));
  // 8xbf16 -> 2xf64
  auto ASliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp00);
  auto ASliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp01);
  auto ASliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp02);
  auto ASliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp03);
  auto ASliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp04);
  auto ASliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp05);
  auto ASliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp06);
  auto ASliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp07);
  auto ASliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp08);
  auto ASliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp09);
  // UZP1 and UZP2
  auto ASliceOp00UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp01UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  auto ASliceOp02UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp03UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  auto ASliceOp04UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp05UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  auto ASliceOp06UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp07UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 0));
  auto ASliceOp08UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  auto ASliceOp09UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  auto A00 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp00UZP1);
  auto A01 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp01UZP2);
  auto A02 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp02UZP1);
  auto A03 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp03UZP2);
  auto A04 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp04UZP1);
  auto A05 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp05UZP2);
  auto A06 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp06UZP1);
  auto A07 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp07UZP2);
  auto A08 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp08UZP1);
  auto A09 =
      rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp09UZP2);

  // Micro Kernel Computation
  // (4xf32, 8xbf16, 8xbf16) -> (4xf32)
  auto BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                              C00, A00, B00);
  auto BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                              C01, A00, B10);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A01, B01);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A01, B11);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A02, B02);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A02, B12);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A03, B03);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A03, B13);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A04, B04);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A04, B14);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A05, B05);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A05, B15);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A06, B06);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A06, B16);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A07, B07);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A07, B17);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A08, B08);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A08, B18);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A09, B09);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A09, B19);

  // Postprocessing C
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA0);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA1);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  // Insert 4xf32 into 8x4xf32
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C00, acc, I64Array{0, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C01, res, I64Array{1, 0}, I64Array{1});

  // Preprocessing C
  // extract 4xf32 from 2x4xf32
  CSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{2, 0},
                                                     I64Array{1, 4}, strides));
  CSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{3, 0},
                                                     I64Array{1, 4}, strides));
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp00);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp01);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);

  // Preprocessing A
  // extract 8xbf16 from 2x40xbf16
  ASliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{2, 0},
                                                     sizes, strides));
  ASliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{2, 8},
                                                     sizes, strides));
  ASliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{2, 16},
                                                     sizes, strides));
  ASliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{2, 24},
                                                     sizes, strides));
  ASliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{2, 32},
                                                     sizes, strides));
  ASliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{3, 0},
                                                     sizes, strides));
  ASliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{3, 8},
                                                     sizes, strides));
  ASliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{3, 16},
                                                     sizes, strides));
  ASliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{3, 24},
                                                     sizes, strides));
  ASliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{3, 32},
                                                     sizes, strides));
  // 8xbf16 -> 2xf64
  ASliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp00);
  ASliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp01);
  ASliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp02);
  ASliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp03);
  ASliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp04);
  ASliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp05);
  ASliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp06);
  ASliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp07);
  ASliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp08);
  ASliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp09);
  // UZP1 and UZP2
  ASliceOp00UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp01UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp02UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp03UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp04UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp05UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp06UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp07UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp08UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp09UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  A00 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp00UZP1);
  A01 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp01UZP2);
  A02 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp02UZP1);
  A03 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp03UZP2);
  A04 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp04UZP1);
  A05 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp05UZP2);
  A06 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp06UZP1);
  A07 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp07UZP2);
  A08 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp08UZP1);
  A09 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp09UZP2);

  // Micro Kernel Computation
  // (4xf32, 8xbf16, 8xbf16) -> (4xf32)
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C00,
                                                         A00, B00);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C01,
                                                         A00, B10);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A01, B01);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A01, B11);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A02, B02);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A02, B12);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A03, B03);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A03, B13);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A04, B04);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A04, B14);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A05, B05);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A05, B15);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A06, B06);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A06, B16);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A07, B07);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A07, B17);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A08, B08);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A08, B18);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A09, B09);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A09, B19);

  // Postprocessing C
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA0);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA1);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  // Insert 4xf32 into 8x4xf32
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C00, res, I64Array{2, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C01, res, I64Array{3, 0}, I64Array{1});

  // Preprocessing C
  // extract 4xf32 from 2x4xf32
  CSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{4, 0},
                                                     I64Array{1, 4}, strides));
  CSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{5, 0},
                                                     I64Array{1, 4}, strides));
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp00);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp01);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);

  // Preprocessing A
  // extract 8xbf16 from 2x40xbf16
  ASliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{4, 0},
                                                     sizes, strides));
  ASliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{4, 8},
                                                     sizes, strides));
  ASliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{4, 16},
                                                     sizes, strides));
  ASliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{4, 24},
                                                     sizes, strides));
  ASliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{4, 32},
                                                     sizes, strides));
  ASliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{5, 0},
                                                     sizes, strides));
  ASliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{5, 8},
                                                     sizes, strides));
  ASliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{5, 16},
                                                     sizes, strides));
  ASliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{5, 24},
                                                     sizes, strides));
  ASliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{5, 32},
                                                     sizes, strides));
  // 8xbf16 -> 2xf64
  ASliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp00);
  ASliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp01);
  ASliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp02);
  ASliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp03);
  ASliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp04);
  ASliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp05);
  ASliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp06);
  ASliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp07);
  ASliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp08);
  ASliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp09);
  // UZP1 and UZP2
  ASliceOp00UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp01UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp02UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp03UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp04UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp05UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp06UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp07UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp08UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp09UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  A00 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp00UZP1);
  A01 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp01UZP2);
  A02 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp02UZP1);
  A03 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp03UZP2);
  A04 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp04UZP1);
  A05 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp05UZP2);
  A06 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp06UZP1);
  A07 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp07UZP2);
  A08 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp08UZP1);
  A09 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp09UZP2);

  // Micro Kernel Computation
  // (4xf32, 8xbf16, 8xbf16) -> (4xf32)
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C00,
                                                         A00, B00);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C01,
                                                         A00, B10);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A01, B01);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A01, B11);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A02, B02);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A02, B12);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A03, B03);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A03, B13);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A04, B04);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A04, B14);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A05, B05);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A05, B15);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A06, B06);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A06, B16);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A07, B07);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A07, B17);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A08, B08);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A08, B18);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A09, B09);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A09, B19);

  // Postprocessing C
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA0);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA1);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  // Insert 4xf32 into 8x4xf32
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C00, res, I64Array{4, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C01, res, I64Array{5, 0}, I64Array{1});

  // Preprocessing C
  // extract 4xf32 from 2x4xf32
  CSliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{6, 0},
                                                     I64Array{1, 4}, strides));
  CSliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, f32x4_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, I64Array{7, 0},
                                                     I64Array{1, 4}, strides));
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp00);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, CSliceOp01);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);

  // Preprocessing A
  // extract 8xbf16 from 2x40xbf16
  ASliceOp00 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{6, 0},
                                                     sizes, strides));
  ASliceOp01 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{6, 8},
                                                     sizes, strides));
  ASliceOp02 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{6, 16},
                                                     sizes, strides));
  ASliceOp03 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{6, 24},
                                                     sizes, strides));
  ASliceOp04 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{6, 32},
                                                     sizes, strides));
  ASliceOp05 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{7, 0},
                                                     sizes, strides));
  ASliceOp06 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{7, 8},
                                                     sizes, strides));
  ASliceOp07 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{7, 16},
                                                     sizes, strides));
  ASliceOp08 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{7, 24},
                                                     sizes, strides));
  ASliceOp09 = rewriter.create<vector::ShapeCastOp>(
      loc, bf16x8_vecTy,
      rewriter.create<vector::ExtractStridedSliceOp>(loc, A, I64Array{7, 32},
                                                     sizes, strides));
  // 8xbf16 -> 2xf64
  ASliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp00);
  ASliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp01);
  ASliceOp02Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp02);
  ASliceOp03Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp03);
  ASliceOp04Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp04);
  ASliceOp05Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp05);
  ASliceOp06Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp06);
  ASliceOp07Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp07);
  ASliceOp08Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp08);
  ASliceOp09Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, ASliceOp09);
  // UZP1 and UZP2
  ASliceOp00UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp01UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp00Bitcast, ASliceOp05Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp02UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp03UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp01Bitcast, ASliceOp06Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp04UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp05UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp02Bitcast, ASliceOp07Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp06UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp07UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp03Bitcast, ASliceOp08Bitcast,
      BoolAttr::get(ctx, 0));
  ASliceOp08UZP1 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 1));
  ASliceOp09UZP2 = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, ASliceOp04Bitcast, ASliceOp09Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 8xbf16
  A00 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp00UZP1);
  A01 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp01UZP2);
  A02 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp02UZP1);
  A03 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp03UZP2);
  A04 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp04UZP1);
  A05 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp05UZP2);
  A06 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp06UZP1);
  A07 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp07UZP2);
  A08 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp08UZP1);
  A09 = rewriter.create<vector::BitCastOp>(loc, bf16x8_vecTy, ASliceOp09UZP2);

  // Micro Kernel Computation
  // (4xf32, 8xbf16, 8xbf16) -> (4xf32)
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C00,
                                                         A00, B00);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy, C01,
                                                         A00, B10);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A01, B01);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A01, B11);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A02, B02);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A02, B12);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A03, B03);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A03, B13);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A04, B04);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A04, B14);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A05, B05);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A05, B15);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A06, B06);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A06, B16);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A07, B07);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A07, B17);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A08, B08);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A08, B18);
  BFMMLA0 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA0, A09, B09);
  BFMMLA1 = rewriter.create<disc_arm_neon_ext::BFMMLAOp>(loc, f32x4_vecTy,
                                                         BFMMLA1, A09, B19);

  // Postprocessing C
  // 4xf32 -> 2xf64
  CSliceOp00Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA0);
  CSliceOp01Bitcast =
      rewriter.create<vector::BitCastOp>(loc, f64x2_vecTy, BFMMLA1);
  // UZP1 and UZP2
  C00UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 1));
  C01UZP = rewriter.create<disc_arm_neon_ext::UZPOp>(
      loc, f64x2_vecTy, CSliceOp00Bitcast, CSliceOp01Bitcast,
      BoolAttr::get(ctx, 0));
  // 2xf64 -> 4xf32
  C00 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C00UZP);
  C01 = rewriter.create<vector::BitCastOp>(loc, f32x4_vecTy, C01UZP);
  // Insert 4xf32 into 8x4xf32
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C00, res, I64Array{6, 0}, I64Array{1});
  res = rewriter.create<vector::InsertStridedSliceOp>(
      loc, C01, res, I64Array{7, 0}, I64Array{1});

  target->replaceAllUsesWith(res);
  results.push_back(res.getOperation());
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
