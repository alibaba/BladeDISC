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
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"

using mlir::memref::DimOp;

namespace mlir {
namespace disc_ral {

int getRowReductionScheduleHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = op->getParentOfType<lmhlo::FusionOp>();
  // Use schedule 2 by default.
  if (!fusion) return DISC_ONE_ROUND_SHUFFLE_ROW_REDUCE;
  IntegerAttr attr =
      fusion->getAttrOfType<IntegerAttr>(kRowReductionScheduleHint);
  if (!attr) return DISC_ONE_ROUND_SHUFFLE_ROW_REDUCE;
  return attr.getInt();
}

int getVectorizationHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = op->getParentOfType<lmhlo::FusionOp>();
  if (!fusion) return 1;
  IntegerAttr attr = fusion->getAttrOfType<IntegerAttr>(kVectorizationHint);
  if (!attr) return 1;
  return attr.getInt();
}

int getThreadPerBlock(Operation* op) {
  int thread_per_block = kThreadsRowReduction;
  if (!op) return thread_per_block;
  lmhlo::FusionOp fusion = op->getParentOfType<lmhlo::FusionOp>();
  if (!fusion) return thread_per_block;
  IntegerAttr attr = fusion->getAttrOfType<IntegerAttr>(kThreadPerBlockHint);
  if (!attr) return thread_per_block;
  return attr.getInt();
}

int getColReductionScheduleHint(Operation* op) {
  assert(op);
  lmhlo::FusionOp fusion = op->getParentOfType<lmhlo::FusionOp>();
  // Use schedule 1 by default.
  if (!fusion) return DISC_TILE_W8_H32;
  IntegerAttr attr =
      fusion->getAttrOfType<IntegerAttr>(kColReductionScheduleHint);
  if (!attr) return DISC_TILE_W8_H32;
  return attr.getInt();
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref) {
  int rank = memref.getType().cast<MemRefType>().getRank();
  Value num_elements;
  num_elements = b.create<arith::ConstantIndexOp>(loc, 1);
  for (int r = 0; r < rank; ++r) {
    auto dim_size = b.create<memref::DimOp>(loc, memref, r);
    num_elements = b.create<arith::MulIOp>(loc, num_elements, dim_size);
  }
  return num_elements;
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op) {
  // only const rank is supported for now
  assert(op->getDialect()->getNamespace() == "lmhlo");
  int num_operands = op->getNumOperands();
  Value result_memref = op->getOperand(num_operands - 1);
  return emitNumElementsComputation(b, loc, result_memref);
}

SmallVector<Value> getShapeValues(OpBuilder* b, Value memref) {
  auto shape = memref.getType().dyn_cast<ShapedType>().getShape();
  int64_t rank = shape.size();
  auto loc = memref.getLoc();

  SmallVector<Value> result;
  for (int i = 0; i < rank; ++i) {
    if (shape[i] == ShapedType::kDynamicSize) {
      result.push_back(b->create<DimOp>(loc, memref, i));
    } else {
      result.push_back(b->create<arith::ConstantIndexOp>(loc, shape[i]));
    }
  }
  return result;
}

Value calcLinearIndex(OpBuilder* b, Location loc, const ValueRange multi_index,
                      const llvm::ArrayRef<Value> shape) {
  assert(multi_index.size() == shape.size());
  return b->create<disc_shape::LinearizeOp>(loc, b->getIndexType(), multi_index,
                                            shape);
}

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index,
                                     const ArrayRef<Value> shape) {
  int rank = shape.size();
  SmallVector<Type> outputTypes(rank, b->getIndexType());
  auto delinearizeOp = b->create<disc_shape::DelinearizeOp>(
      loc, outputTypes, linear_index, shape);

  SmallVector<Value> results;
  for (Value result : delinearizeOp->getResults()) results.push_back(result);
  return results;
}

Value getDimSizeValue(OpBuilder* b, Value memref, int dim) {
  auto memref_ty = memref.getType().dyn_cast<ShapedType>();
  auto loc = memref.getLoc();
  assert(memref_ty && memref_ty.getRank() > dim);
  auto dim_size = memref_ty.getDimSize(dim);
  if (dim_size == ShapedType::kDynamicSize) {
    return b->create<DimOp>(loc, memref, dim);
  } else {
    return b->create<arith::ConstantIndexOp>(loc, dim_size);
  }
}

Value mayConvertToIndexType(Value val, OpBuilder* b, Location loc) {
  if (val.getType().isIndex()) return val;
  return b->create<arith::IndexCastOp>(loc, val, b->getIndexType());
}

Value mayConvertToIntegerType(Value val, OpBuilder* b, Location loc) {
  if (val.getType().isInteger(32) || val.getType().isInteger(64)) return val;
  return b->create<arith::IndexCastOp>(loc, val, b->getI64Type());
}

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index, Value memref) {
  SmallVector<Value> shape_vec = getShapeValues(b, memref);
  return calcMultiDimIndex(b, loc, linear_index, shape_vec);
}

Value CastMemRefTo(OpBuilder& b, Location loc, Value from, Type toType,
                   ValueRange toShape) {
  int64_t rank = toShape.size();
  auto memrefTy = toType.cast<MemRefType>();
  auto getIntAttr = [&](int64_t val) {
    return b.getIntegerAttr(b.getIndexType(), val);
  };

  SmallVector<OpFoldResult> foldSizes;
  for (int i = 0; i < rank; ++i) {
    if (memrefTy.getDimSize(i) == ShapedType::kDynamicSize) {
      foldSizes.push_back(toShape[i]);
    } else {
      foldSizes.push_back(getIntAttr(memrefTy.getDimSize(i)));
    }
  }

  Value dynamicStride;
  int64_t staticStride = 1;
  SmallVector<OpFoldResult> foldStrides;
  for (int i = rank - 1; i >= 0; --i) {
    if (staticStride != ShapedType::kDynamicSize) {
      foldStrides.push_back(getIntAttr(staticStride));
      if (memrefTy.getDimSize(i) == ShapedType::kDynamicSize) {
        dynamicStride = b.create<arith::ConstantIndexOp>(loc, staticStride);
        staticStride = ShapedType::kDynamicSize;
      } else {
        staticStride *= memrefTy.getDimSize(i);
      }
    } else {
      foldStrides.push_back(dynamicStride);
    }
    if (dynamicStride) {
      dynamicStride = b.create<arith::MulIOp>(loc, dynamicStride, toShape[i]);
    }
  }

  OpFoldResult zero{getIntAttr(0)};
  auto reversedStrides = llvm::to_vector<4>(llvm::reverse(foldStrides));
  auto castOp = b.create<memref::ReinterpretCastOp>(loc, memrefTy, from, zero,
                                                    foldSizes, reversedStrides);
  return castOp;
}

using scf::IfOp;
using scf::ParallelOp;

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> init_values) {
  auto par_op = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, init_values,
                                          /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(par_op.getBody());
  vars.append(par_op.getInductionVars().begin(),
              par_op.getInductionVars().end());
  return par_op;
}

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(%arg4*tileSize[0], %arg2-%i0)
///                                          min(%arg5*tileSize[1], %arg3-%i1))
///                                      step (%arg4, %arg5)
///
/// or, when with-inbound-check is true, into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (%arg4*tileSize[0],
///                                          %arg5*tileSize[1])
///                                      step (%arg4, %arg5)
///        %inbound = (%j0 * %arg4 + %i0 < %arg2) &&
///                   (%j1 * %arg5 + %i1 < %arg3)
///        scf.if (%inbound)
///          ....
///
/// where the uses of %i0 and %i1 in the loop body are replaced by
/// %i0 + j0 and %i1 + %j1.
//
/// The old loop is replaced with the new one.
std::pair<ParallelOp, ParallelOp> tileParallelLoop(ParallelOp op,
                                                   ArrayRef<int64_t> tileSizes,
                                                   bool withInboundCheck) {
  OpBuilder b(op);
  auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  SmallVector<Value, 2> tileSizeConstants;
  tileSizeConstants.reserve(op.getUpperBound().size());
  for (size_t i = 0, end = op.getUpperBound().size(); i != end; ++i) {
    if (i < tileSizes.size())
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), tileSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create the outer loop with adjusted steps.
  SmallVector<Value, 2> newSteps;
  newSteps.reserve(op.getStep().size());
  for (auto step : llvm::zip(op.getStep(), tileSizeConstants)) {
    newSteps.push_back(b.create<arith::MulIOp>(op.getLoc(), std::get<0>(step),
                                               std::get<1>(step)));
  }
  auto outerLoop = b.create<ParallelOp>(op.getLoc(), op.getLowerBound(),
                                        op.getUpperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, b.getContext()),
       getAffineDimExpr(/*position=*/1, b.getContext()) -
           getAffineDimExpr(/*position=*/2, b.getContext())},
      b.getContext());

  // Create the inner loop with adjusted bounds.
  SmallVector<Value, 2> newBounds;
  newBounds.reserve(op.getUpperBound().size());
  bool needInboundCheck = false;
  for (auto dim :
       llvm::zip(outerLoop.getLowerBound(), outerLoop.getUpperBound(),
                 outerLoop.getStep(), outerLoop.getInductionVars(),
                 op.getStep(), tileSizeConstants)) {
    Value lowerBound, upperBound, newStep, iv, step, tileSizeConstant;
    std::tie(lowerBound, upperBound, newStep, iv, step, tileSizeConstant) = dim;
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
    auto tileSize =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp())
            .getValue()
            .cast<IntegerAttr>()
            .getInt();
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(
          upperBoundConstant.getValue().cast<IntegerAttr>().getInt() -
              lowerBoundConstant.getValue().cast<IntegerAttr>().getInt(),
          stepConstant.getValue().cast<IntegerAttr>().getInt());
      if (numIterations % tileSize == 0) {
        newBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (withInboundCheck) {
      newBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBounds.push_back(
        b.create<AffineMinOp>(op.getLoc(), b.getIndexType(), minMap,
                              ValueRange{newStep, upperBound, iv}));
  }
  auto innerLoop = b.create<ParallelOp>(
      op.getLoc(), SmallVector<Value, 2>(newBounds.size(), zero), newBounds,
      op.getStep());

  if (withInboundCheck && needInboundCheck) {
    b.setInsertionPointToStart(innerLoop.getBody());
    // Insert in-bound check
    Value inbound =
        b.create<arith::ConstantIntOp>(op.getLoc(), 1, /*bitWidth*/ 1);
    for (auto dim :
         llvm::zip(outerLoop.getUpperBound(), outerLoop.getInductionVars(),
                   innerLoop.getInductionVars(), innerLoop.getStep())) {
      Value outerUpperBound, outerIV, innerIV, innerStep;
      std::tie(outerUpperBound, outerIV, innerIV, innerStep) = dim;
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv < %outer_upper_bound)
      Value index = b.create<arith::AddIOp>(
          op.getLoc(), b.create<arith::MulIOp>(op.getLoc(), innerIV, innerStep),
          outerIV);
      Value dimInbound = b.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ult, index, outerUpperBound);
      inbound = b.create<arith::AndIOp>(op.getLoc(), inbound, dimInbound);
    }
    auto ifInbound = b.create<IfOp>(op.getLoc(),
                                    /*resultTypes*/ ArrayRef<Type>{}, inbound,
                                    /*hasElseRegion*/ false);
    ifInbound.getThenRegion().takeBody(op.getLoopBody());
    Block& thenBlock = ifInbound.getThenRegion().front();
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::enumerate(llvm::zip(innerLoop.getInductionVars(),
                                              outerLoop.getInductionVars()))) {
      arith::AddIOp newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs.value()), std::get<1>(ivs.value()));
      thenBlock.getArgument(ivs.index())
          .replaceAllUsesExcept(newIndex, newIndex);
    }
    thenBlock.eraseArguments(llvm::to_vector<4>(
        llvm::seq((unsigned)0, thenBlock.getNumArguments())));
  } else {
    innerLoop.getLoopBody().takeBody(op.getLoopBody());
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::zip(innerLoop.getInductionVars(),
                              outerLoop.getInductionVars())) {
      Value inner_index = std::get<0>(ivs);
      arith::AddIOp newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs), std::get<1>(ivs));
      inner_index.replaceAllUsesExcept(newIndex, newIndex);
    }
  }

  op.erase();
  return std::make_pair(outerLoop, innerLoop);
}

int initReductionTileSizeOnCPU() {
  const char* env = getenv("DISC_REDUCTION_TILE_SIZE");
  if (!env) return kReductionTileSizeOnCPU;
  return std::stoi(env);
}

int getReductionTileSizeOnCPU() {
  static int size = initReductionTileSizeOnCPU();
  return size;
}

}  // namespace disc_ral
}  // namespace mlir
