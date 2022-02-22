/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements the logic to lower some specific ops to external library
// calls.
//
// Here the external function is model by a `disc_ral.dispatch` op. We use
// `disc_ral.dispatch` to serve as a unified entrance of disc external
// calls due to following reasons.
// - `disc_ral.dispatch` ensures that the first argument is always the
//   `disc_ral.context`
// - `disc_ral.dispatch` simplifies the logic to handle different instantiations
//   of one op for different devices and different element types. For example,
//   we may have GEMM ops with different element types.

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/IR/custom_call_base.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_ral_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

namespace {

using lmhlo::ConvOp;
using lmhlo::CopyOp;
using lmhlo::DotGeneralOp;
using lmhlo::DynamicConvOp;
using lmhlo::DynamicReshapeOp;
using lmhlo::FusionOp;
using lmhlo::LmhloOp;
using lmhlo::ReshapeOp;
using lmhlo_disc::CustomCallOp;
using lmhlo_disc::D2HOp;
using lmhlo_disc::H2DOp;

// Suppose that the first argument of the function is the ctx value
Value GetContextValueFromFunctionArguments(Operation* op) {
  Value ctx;
  if (auto func = op->getParentOfType<FuncOp>()) {
    if (func.getArgument(0).getType().isa<RalExecutionContextType>()) {
      return func.getArgument(0);
    }
    op->emitError() << "Argument#0 must be RalExecutionContextType.";
  }
  return ctx;
}

// Currently we only use a single stream. Re-visit this if necessary.
Value GetDefaultStreamHandle(Operation* op, PatternRewriter& rewriter) {
  Location loc = op->getLoc();
  MLIRContext* ctx = rewriter.getContext();
  Type llvm_int32_type = IntegerType::get(ctx, 32);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  Type pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Value stream_idx = rewriter.create<LLVM::IntToPtrOp>(loc, pointer_type, zero);
  return stream_idx;
}

// Insert a sync on stream call.
void InsertSyncOnStream(Operation* op, Value ctx, Value stream_handle,
                        PatternRewriter& rewriter) {
  Location loc = op->getLoc();
  rewriter.create<DispatchOp>(loc, llvm::None, ctx, stream_handle,
                              "sync_on_stream", false, "gpu");
}

// Converting:
//   %output = disc_ral.recv_input(ctx, input_idx)
//     to
//   %output = disc_ral.dispatch(ctx, input_idx) {call_target_name =
//   "ral_recv_input", backend_config = "cpu"}
struct RecvInputOpConvertor : public OpRewritePattern<RecvInputOp> {
  using OpRewritePattern<RecvInputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecvInputOp op,
                                PatternRewriter& rewriter) const override {
    auto operands = op.getOperands();
    rewriter.replaceOpWithNewOp<DispatchOp>(op, op.getType(), operands.front(),
                                            operands.drop_front(),
                                            "ral_recv_input", false, "cpu");
    return success();
  }
};

// Converting:
//   disc_ral.send_output(ctx, output_idx, output)
//     to
//   disc_ral.dispatch(ctx, output_idx, output) {call_target_name =
//   "ral_send_output", backend_config = "cpu"}
struct SendOutputOpConvertor : public OpRewritePattern<SendOutputOp> {
  using OpRewritePattern<SendOutputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SendOutputOp op,
                                PatternRewriter& rewriter) const override {
    auto operands = op.getOperands();
    rewriter.replaceOpWithNewOp<DispatchOp>(op, llvm::None, operands.front(),
                                            operands.drop_front(),
                                            "ral_send_output", false, "cpu");
    return success();
  }
};

// Converting:
//   xxxDialect.xxxOp(operands...)
//     to
//   ctx = getCtxFromFirstArgumentOfFunction()
//   stream_value = getDefaultGpuStream()
//   newOperands = {stream_value, operands...};
//   disc_ral.dispatch(ctx, newOperands) {call_target_name =
//     "xxx", backend_config = "gpu"}
//   disc_ral.dispatch(ctx, stream_value) {call_target_name =
//     "sync_on_stream", backend_config = "gpu"}
template <typename OpTy>
struct GpuCopyOpConvertor : public OpRewritePattern<OpTy> {
  GpuCopyOpConvertor(MLIRContext* context, StringRef target)
      : OpRewritePattern<OpTy>::OpRewritePattern(context) {
    this->target_ = target;
  }

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      return op->emitOpError(
          "the first argument of the function is not ral context type");
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);

    SmallVector<Value, 4> newOperands{stream_handle};
    for (Value operand : op.getOperands()) {
      newOperands.push_back(operand);
    }

    auto newOp = rewriter.create<DispatchOp>(
        op.getLoc(), llvm::None, ctx, newOperands, target_, false, "gpu");
    // TODO(disc): Re-visit this is necessary.
    // TODO(disc): add a pass to merge sync_on_stream call.
    InsertSyncOnStream(op, ctx, stream_handle, rewriter);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

 private:
  StringRef target_;
};

struct DotGeneralOpConvertor : public OpRewritePattern<DotGeneralOp> {
  using OpRewritePattern<DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      return op->emitOpError()
             << "the first argument of the function is not ral context type";
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);

    SmallVector<Value, 4> newOperands{stream_handle};
    for (Value operand : op.getOperands()) {
      newOperands.push_back(operand);
    }

    auto dot_dimension_attr = op.dot_dimension_numbers();
    auto lhs_batching_dims = dot_dimension_attr.getLhsBatchingDimensions();
    auto rhs_batching_dims = dot_dimension_attr.getRhsBatchingDimensions();

    if (lhs_batching_dims.size() != rhs_batching_dims.size()) {
      return op.emitOpError() << "unmatched batch dims size.";
    }

    int rank = op.getOperand(0).getType().cast<MemRefType>().getRank();
    for (auto&& z : llvm::zip(lhs_batching_dims, rhs_batching_dims)) {
      if ((std::get<0>(z) >= rank - 2) || (std::get<1>(z) >= rank - 2)) {
        return op.emitOpError() << "unsupported batch dims.";
      }
    }

    auto lhs_contracting_dims =
        dot_dimension_attr.getLhsContractingDimensions();
    auto rhs_contracting_dims =
        dot_dimension_attr.getRhsContractingDimensions();
    // TODO: support multi-dim contracting
    if ((lhs_contracting_dims.size() != 1) ||
        (rhs_contracting_dims.size() != 1)) {
      return op.emitOpError()
             << "DotGeneralOp only supports 1 dimensional contracting.";
    }
    if (((lhs_contracting_dims[0] != rank - 1) &&
         (lhs_contracting_dims[0] != rank - 2)) ||
        ((rhs_contracting_dims[0] != rank - 1) &&
         (rhs_contracting_dims[0] != rank - 2))) {
      return op.emitOpError()
             << "DotGeneral only support contracting through the last "
                "two dimensions.";
    }

    bool tp_lhs = (lhs_contracting_dims[0] == (rank - 2));
    bool tp_rhs = (rhs_contracting_dims[0] == (rank - 1));

    newOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), tp_lhs, /*bitWidth*/ 1));
    newOperands.push_back(rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), tp_rhs, /*bitWidth*/ 1));

    bool on_gpu = placement_utils::isGpuMemRef(op->getOperand(2));
    rewriter.replaceOpWithNewOp<DispatchOp>(op, llvm::None, ctx, newOperands,
                                            "ral_gemm", false,
                                            on_gpu ? "gpu" : "cpu");

    return success();
  }
};

template <typename OpTy>
Value GetConvMetadata(OpTy op, PatternRewriter& rewriter) {
  // Metadata:
  //   - input layput: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - kernel layout: each field for one dimension. The order is:
  //     * in_channel, out_channel, spatial dimensions
  //   - output layout: each field for one dimension. The order is:
  //     * batch, channel, spatial dimensions
  //   - strides: each filed for one spatial dimension.
  //   - dilations: each filed for one spatial dimension.
  Location loc = op.getLoc();
  Type field_type = rewriter.getI32Type();
  int rank = op.output().getType().template dyn_cast<ShapedType>().getRank();
  int num_spatial_dims = rank - 2;
  int num_metadata_fields = rank * 3 + (rank - 2) * 2;
  Value metadata_value = rewriter.create<memref::AllocaOp>(
      loc, MemRefType::get(
               {num_metadata_fields}, field_type, MemRefLayoutAttrInterface(),
               StringAttr::get(op->getContext(), placement_utils::kCpu)));
  std::vector<int64_t> fields;
  auto dimension_numbers = op.dimension_numbers();
  // input layout
  fields.push_back(dimension_numbers.getInputBatchDimension());
  fields.push_back(dimension_numbers.getInputFeatureDimension());
  auto input_spatial_dimensions = dimension_numbers.getInputSpatialDimensions();
  fields.insert(fields.end(), input_spatial_dimensions.begin(),
                input_spatial_dimensions.end());
  // kernel layout
  fields.push_back(dimension_numbers.getKernelInputFeatureDimension());
  fields.push_back(dimension_numbers.getKernelOutputFeatureDimension());
  auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  fields.insert(fields.end(), kernel_spatial_dimensions.begin(),
                kernel_spatial_dimensions.end());
  // output layout
  fields.push_back(dimension_numbers.getOutputBatchDimension());
  fields.push_back(dimension_numbers.getOutputFeatureDimension());
  auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  fields.insert(fields.end(), output_spatial_dimensions.begin(),
                output_spatial_dimensions.end());
  // strides
  auto window_strides = disc_ral::ConvertDenseIntAttr(op.window_strides());
  fields.insert(fields.end(), window_strides.begin(), window_strides.end());
  // rhs_dilation
  auto rhs_dilation = disc_ral::ConvertDenseIntAttr(op.rhs_dilation());
  fields.insert(fields.end(), rhs_dilation.begin(), rhs_dilation.end());

  for (auto&& en : llvm::enumerate(fields)) {
    Value value =
        rewriter.create<arith::ConstantIntOp>(loc, en.value(), field_type);
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, en.index());
    SmallVector<Value, 1> ivs(1, offset);
    rewriter.create<memref::StoreOp>(loc, value, metadata_value, ivs);
  }

  return metadata_value;
}

struct ConvConverter : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  Value GetPadding(ConvOp op, PatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    Type field_type = rewriter.getI32Type();
    int rank = op.output().getType().template cast<ShapedType>().getRank();
    int num_metadata_fields = (rank - 2) * 2;
    Value metadata_value = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get(
                 {num_metadata_fields}, field_type, MemRefLayoutAttrInterface(),
                 StringAttr::get(op->getContext(), placement_utils::kCpu)));
    // padding
    auto padding = disc_ral::ConvertDenseIntAttr(op.padding());
    for (auto&& en : llvm::enumerate(padding)) {
      Value value =
          rewriter.create<arith::ConstantIntOp>(loc, en.value(), field_type);
      Value offset = rewriter.create<arith::ConstantIndexOp>(loc, en.index());
      SmallVector<Value, 1> ivs(1, offset);
      rewriter.create<memref::StoreOp>(loc, value, metadata_value, ivs);
    }
    return metadata_value;
  }

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      return op->emitOpError()
             << "the first argument of the function is not ral context type.";
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);
    SmallVector<Value, 4> newOperands{stream_handle};

    // input
    newOperands.push_back(op.getOperand(0));
    // kernel
    newOperands.push_back(op.getOperand(1));
    // padding
    newOperands.push_back(GetPadding(op, rewriter));
    // output
    newOperands.push_back(op.getOperand(2));
    // input & kernel & output layouts, strides and dilation
    newOperands.push_back(GetConvMetadata(op, rewriter));

    bool on_gpu = placement_utils::isGpuMemRef(op->getOperand(2));
    rewriter.replaceOpWithNewOp<DispatchOp>(op, llvm::None, ctx, newOperands,
                                            "ral_conv", false,
                                            on_gpu ? "gpu" : "cpu");
    return success();
  }
};

struct DynamicConvConverter : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern<DynamicConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      op->emitOpError()
          << "the first argument of the function is not ral context type.";
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);
    SmallVector<Value, 4> newOperands{stream_handle};

    for (Value operand : op.getOperands()) {
      newOperands.push_back(operand);
    }

    // input & kernel & output layouts, strides and dilation
    newOperands.push_back(GetConvMetadata(op, rewriter));

    bool on_gpu = placement_utils::isGpuMemRef(op->getOperand(3));
    rewriter.replaceOpWithNewOp<DispatchOp>(op, llvm::None, ctx, newOperands,
                                            "ral_conv", false,
                                            on_gpu ? "gpu" : "cpu");
    return success();
  }
};

// Return null is not a copy-removable copy-like ops, otherwise return
// the copied result value.
Value getCopyRemovableResult(Operation* op) {
  // Not removable if inside a fusion.
  if (op->getParentOfType<FusionOp>() != nullptr) return {};

  // TODO(disc): add cpu copy removal support.
  if (isa<ReshapeOp, DynamicReshapeOp, CopyOp>(op)) {
    Value result = cast<LmhloOp>(op).getResultBuffer();
#ifdef TAO_CPU_ONLY
    if (!IsSmallCpuBuffer(result) && !IsSmallCpuBuffer(op->getOperand(0)))
      return result;
#else
    if (placement_utils::isGpuMemRef(result)) return result;
#endif
  }

  return {};
}

Value getRootMemRefIfSafe(Value memref) {
  Value rootMemRef = memref;
  DenseSet<Operation*> knownSafeOpSet;
  while (auto view = dyn_cast_or_null<ViewLikeOpInterface>(
             rootMemRef.getDefiningOp())) {
    knownSafeOpSet.insert(view);
    if (rootMemRef != memref) {
      for (Operation* user : rootMemRef.getUsers()) {
        if (isa<memref::DimOp>(user)) continue;
        if (knownSafeOpSet.contains(user)) continue;
        // not safe to elide the copy since the underline view have unsafe
        // users.
        return {};
      }
    }
    rootMemRef = view->getOperand(0);
  }
  if (rootMemRef != memref) {
    for (Operation* user : rootMemRef.getUsers()) {
      // Only the final rootMemRef is allowed to have dealloc op.
      // Return match failure for other cases.
      if (isa<memref::DimOp, memref::DeallocOp>(user)) continue;
      if (knownSafeOpSet.contains(user)) continue;
      // not safe to elide the copy since the underline view have unsafe users.
      return {};
    }
  }

  return rootMemRef;
}

// Converting:
//  lmhlo.xxxOp(%from, %to)
//    to
//  ctx = getCtxFromFirstArgumentOfFunction()
//  stream_value = getDefaultGpuStream()
//  %target_shape = shape_of(%to)
//  newOperands = {stream_value, %from, %target_shape}
//  %newTo = disc_ral.dispatch(ctx, newOperands) {call_target_name =
//     "inc_ref", backend_config = "gpu"}
//  replace not shape-only users of %to using %newTo
template <typename OpTy>
struct CopyLikeOpConvertor : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Value result = getCopyRemovableResult(op);
    if (!result) return failure();
    Value rootResult = getRootMemRefIfSafe(result);
    if (!rootResult) return failure();

    // users of original result that are placed after this op.
    SmallVector<Operation*, 4> users;
    bool hasDeallocOpUser = false;
    Block* block = op->getBlock();
    for (Operation* user : result.getUsers()) {
      if (user == op) continue;
      // shapeOf op should already be lowered before this pass.
      if (isa<memref::DimOp>(user)) continue;
      Operation* ancestor = block->findAncestorOpInBlock(*user);
      // not safe to remove the copy in such case.
      if (!ancestor) return failure();
      if (!op->isBeforeInBlock(ancestor)) return failure();
      users.push_back(user);
      if (isa<memref::DeallocOp>(user)) {
        // Not able to handle multiple dealloc case
        if (hasDeallocOpUser) return failure();
        hasDeallocOpUser = true;
      }
    }
    Operation* rootResultDeallocOp = nullptr;
    if (!hasDeallocOpUser && (rootResult != result)) {
      for (Operation* user : rootResult.getUsers()) {
        if (isa<memref::DeallocOp>(user)) {
          // Not able to handle multiple dealloc case.
          if (rootResultDeallocOp) return failure();
          // Not safe to insert deallocOp for the newValue in such case.
          if (!op->isBeforeInBlock(user)) return failure();
          rootResultDeallocOp = user;
        }
      }
    }

    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      return op->emitError(
          "the first argument of the function is not ral context type");
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);

    Location loc = op.getLoc();
    Type shapeIndexType = rewriter.getIndexType();
    auto targetType = result.getType().cast<MemRefType>();
    auto shapeType = MemRefType::get(
        {targetType.getRank()}, shapeIndexType, targetType.getLayout(),
        StringAttr::get(op->getContext(), placement_utils::kCpu));
    Value targetShape = rewriter.create<memref::AllocaOp>(loc, shapeType);
    SmallVector<Value> dimSizes;
    for (int i = 0; i < targetType.getRank(); ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value dimSize = rewriter.create<memref::DimOp>(loc, result, idx);
      dimSizes.push_back(dimSize);
      rewriter.create<memref::StoreOp>(loc, dimSize, targetShape, idx);
    }

    SmallVector<Value, 4> newOperands{stream_handle, op->getOperand(0),
                                      targetShape};

    auto dispatchOp = rewriter.create<DispatchOp>(
        loc, targetType, ctx, newOperands, "inc_ref", false,
        placement_utils::isGpuMemRef(result) ? "gpu" : "cpu");

    Value newValue = CastMemRefTo(rewriter, loc, dispatchOp->getResult(0),
                                  targetType, dimSizes);

    // replace non-shape-consumer users of original result
    for (Operation* user : users) user->replaceUsesOfWith(result, newValue);
    // Insert a dealloc op to handle following case:
    //  origin:
    //   %0 = memref.alloc() : memref<10xf32>
    //   %1 = memref.cast(%0) : memref<?xf32>
    //   lmhlo.copy(%arg0, %1)
    //   use(%1)
    //   ...
    //   memref.dealloc(%0)
    //  after changing:
    //   %0 = memref.alloc() : memref<10xf32>
    //   %1 = memref.cast(%0) : memref<?xf32>
    //   %2 = disc_ral.dispatch(..., "inc_ref", ...)
    //   use(%2)
    //   ...
    //   memref.dealloc(%2)
    //   memref.dealloc(%0)
    if (rootResultDeallocOp) {
      rewriter.setInsertionPoint(rootResultDeallocOp);
      rewriter.create<memref::DeallocOp>(loc, newValue);
    }

    // remove original op
    rewriter.eraseOp(op);
    return success();
  }
};

struct CustomCallOpConvertor : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern<CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    Value ctx = GetContextValueFromFunctionArguments(op);
    if (!ctx) {
      return op.emitOpError()
             << "the first argument of the function is not ral context type";
    }
    Value stream_handle = GetDefaultStreamHandle(op, rewriter);

    StringRef target_name = op.call_target_name();
    auto lower_func =
        mhlo_disc::CustomCallRegistry::Global().FindLowerToLibraryCallFunc(
            target_name.str());
    if (!lower_func) {
      return op.emitOpError()
             << "custom call " << target_name
             << " is not implemented in lowering to library calls.";
    }
    return lower_func(op, rewriter, ctx, stream_handle);
  }
};

struct DiscLowerToLibraryCallPass
    : public DiscLowerToLibraryCallPassBase<DiscLowerToLibraryCallPass> {
  using DiscLowerToLibraryCallPassBase<
      DiscLowerToLibraryCallPass>::DiscLowerToLibraryCallPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    // clang-format off
    patterns.insert<
      CopyLikeOpConvertor<CopyOp>,
      CopyLikeOpConvertor<DynamicReshapeOp>,
      CopyLikeOpConvertor<ReshapeOp>,
      CustomCallOpConvertor,
      DotGeneralOpConvertor,
      RecvInputOpConvertor,
      ConvConverter,
      DynamicConvConverter,
      SendOutputOpConvertor
    >(context);
    // clang-format on

    // GPU copy related ops
    patterns.insert<GpuCopyOpConvertor<H2DOp>>(context, "h2d");
    patterns.insert<GpuCopyOpConvertor<D2HOp>>(context, "d2h");

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscLowerToLibraryCallPass() {
  return std::make_unique<DiscLowerToLibraryCallPass>();
}

}  // namespace disc_ral
}  // namespace mlir
