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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"

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

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation* createLinalgCopyOp(OpBuilder& b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {}) {
  auto memrefTypeFrom = from.getType().dyn_cast<MemRefType>();
  auto memrefTypeTo = to.getType().dyn_cast<MemRefType>();
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<StringRef> iteratorTypes(memrefTypeTo.getRank(),
                                       getParallelIteratorTypeName());
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::makeArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder& b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
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

}  // namespace transform_dialect

void registerTransformDialectCommonExtension(DialectRegistry& registry) {
  registry
      .addExtensions< ::mlir::disc_ral::transform_dialect::CommonExtensions>();
}

}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.cc.inc"
