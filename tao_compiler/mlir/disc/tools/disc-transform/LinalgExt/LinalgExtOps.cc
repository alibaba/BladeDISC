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

#include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

Value getDimValue(OpBuilder& builder, Location loc, Value v, int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder& builder, Location loc, Value v, int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

SmallVector<OpFoldResult> getDims(OpBuilder& builder, Location loc,
                                  Value shapedTypeValue) {
  return llvm::to_vector(llvm::map_range(
      llvm::seq<int64_t>(
          0, shapedTypeValue.getType().cast<ShapedType>().getRank()),
      [&](int64_t dim) { return getDim(builder, loc, shapedTypeValue, dim); }));
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

static int64_t ceilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

/// Custom builder methods for pack ops.
void MultiLevelPackOp::build(OpBuilder& builder, OperationState& state,
                             Value source, Value output,
                             ArrayRef<int64_t> tileLevels,
                             ArrayRef<int64_t> tileSizes,
                             ArrayRef<int64_t> permutation,
                             Optional<Value> paddingValue) {
  SmallVector<int64_t> permutationVec;
  int64_t expectedResultRank =
      MultiLevelPackOp::getExpectedResultRank(tileLevels);
  if (expectedResultRank > 0 && permutation.empty()) {
    permutationVec = llvm::to_vector<>(
        llvm::seq(static_cast<int64_t>(0), expectedResultRank));
    permutation = permutationVec;
  }
  ShapedType resultType = getPackedType(source.getType().cast<ShapedType>(),
                                        tileLevels, tileSizes, permutation);
  build(builder, state, resultType, source, output,
        builder.getI64ArrayAttr(tileLevels), builder.getI64ArrayAttr(tileSizes),
        builder.getI64ArrayAttr(permutation),
        (paddingValue ? paddingValue.value() : nullptr));
}

/* static */ ShapedType MultiLevelPackOp::getPackedType(
    ShapedType inputType, ArrayRef<int64_t> tileLevels,
    ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> permutation) {
  int expectedResultRank = MultiLevelPackOp::getExpectedResultRank(tileLevels);
  SmallVector<int64_t> tiledShape(expectedResultRank, ShapedType::kDynamicSize);
  int tileSizeIdx = 0;
  int tiledDimIdx = 0;
  for (int dimIdx = 0; dimIdx < inputType.getRank(); ++dimIdx) {
    int64_t dimSize = inputType.getShape()[dimIdx];
    int64_t level = tileLevels[dimIdx];
    int lastTileSize = 1;
    for (int localTiledDimIdx = level; localTiledDimIdx > 0;
         --localTiledDimIdx) {
      int64_t tileSize = tileSizes[tileSizeIdx + localTiledDimIdx - 1];
      tiledShape[tiledDimIdx + localTiledDimIdx] =
          ceilDiv(tileSize, lastTileSize);
      lastTileSize = tileSize;
    }
    if (dimSize != ShapedType::kDynamicSize)
      tiledShape[tiledDimIdx] = ceilDiv(dimSize, lastTileSize);
    tileSizeIdx += level;
    tiledDimIdx += 1 + level;
  }

  if (!permutation.empty()) {
    tiledShape = interchange<int64_t>(tiledShape, permutation, /*offset=*/0);
  }

  return TypeSwitch<ShapedType, ShapedType>(inputType)
      .Case<RankedTensorType>([&](auto shapedType) {
        return RankedTensorType::get(tiledShape, shapedType.getElementType());
      })
      .Case<MemRefType>([&](auto shapedType) {
        return MemRefType::get(tiledShape, shapedType.getElementType());
      })
      .Default([&](Type t) {
        assert(false && "unexpected type");
        return nullptr;
      });
}

LogicalResult MultiLevelPackOp::verify() {
  Operation* op = getOperation();
  int64_t inputRank = getInputRank();
  if (inputRank != getTileLevels().size()) {
    return op->emitError("mismatch input rank and the size of tile_levels ")
           << inputRank << " vs " << getTileLevels().size() << "\n";
  }
  int64_t expectedResultRank = getExpectedResultRank();
  if (expectedResultRank != getPermutation().size()) {
    return op->emitError(
               "mismatch expected output rank and the size of permutation ")
           << expectedResultRank << " vs " << getPermutation().size() << "\n";
  }
  if (expectedResultRank != getOutputRank()) {
    return op->emitError(
               "mismatch expected output rank and the rank of the output "
               "operand ")
           << expectedResultRank << " vs " << getOutputRank() << "\n";
  }

  auto sortedPermutation = getPermutationVec();
  llvm::sort(sortedPermutation);
  if (!sortedPermutation.empty() &&
      (sortedPermutation[0] != 0 ||
       sortedPermutation[expectedResultRank - 1] != expectedResultRank - 1)) {
    return op->emitError("not a valid permutation setting\n");
  }

  auto tileLevels = getTileLevelsVec();
  auto tileSizes = getTileSizesVec();
  auto permutation = getPermutationVec();
  auto expectedType =
      getPackedType(getInputType(), tileLevels, tileSizes, permutation);
  if (!expectedType) {
    return op->emitError("failed to infer the packed type\n");
  }
  if (expectedType != getOutputType()) {
    return op->emitError(
               "mismatch expected output type and actual output type ")
           << expectedType << " vs " << getOutputType() << "\n";
  }

  return success();
}

void MultiLevelPackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

SmallVector<OpFoldResult> MultiLevelPackOp::getResultShape(
    OpBuilder& builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<int64_t> tileLevels, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> permutation) {
  int expectedResultRank = getExpectedResultRank(tileLevels);
  SmallVector<OpFoldResult> resultDims(expectedResultRank);

  auto const2IndexAttr = [&](int64_t val) {
    return IntegerAttr::get(builder.getIndexType(), val);
  };

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);

  int tileSizeIdx = 0;
  int resultDimIdx = 0;
  for (int dimIdx = 0; dimIdx < sourceDims.size(); ++dimIdx) {
    int64_t level = tileLevels[dimIdx];
    OpFoldResult dimSize = sourceDims[dimIdx];
    OpFoldResult lastTileSize = const2IndexAttr(1);
    for (int localResultDimIdx = level; localResultDimIdx > 0;
         --localResultDimIdx) {
      OpFoldResult tileSize =
          const2IndexAttr(tileSizes[tileSizeIdx + localResultDimIdx - 1]);
      resultDims[resultDimIdx + localResultDimIdx] =
          makeComposedFoldedAffineApply(builder, loc, ceilDivExpr,
                                        {tileSize, lastTileSize});
      lastTileSize = tileSize;
    }
    resultDims[resultDimIdx] = makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr, {dimSize, lastTileSize});
    tileSizeIdx += level;
    resultDimIdx += 1 + level;
  }

  if (!permutation.empty()) {
    resultDims =
        interchange<OpFoldResult>(resultDims, permutation, /*offset=*/0);
  }
  return resultDims;
}

SmallVector<OpFoldResult> MultiLevelPackOp::getResultShape(OpBuilder& builder) {
  auto tileLevels = getTileLevelsVec();
  auto tileSizes = getTileSizesVec();
  auto permutation = getPermutationVec();
  return getResultShape(builder, getLoc(),
                        getDims(builder, getLoc(), getInput()), tileLevels,
                        tileSizes, permutation);
}

LogicalResult MultiLevelPackOp::reifyResultShapes(
    OpBuilder& builder, ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = getValueOrCreateConstantIndexOp(
      builder, getLoc(), getResultShape(builder));
  return success();
}

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgExtOp> {
  using OpInterfaceRewritePattern<LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgExtOp op,
                                PatternRewriter& rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand* opOperand) {
          if (opOperand->get().isa<BlockArgument>()) return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand) return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand* opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand* opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Add the other operands.
    for (OpOperand* opOperand : op.getNonInputOrOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Clone op.
    Operation* newOp =
        cast<DestinationStyleOpInterface>(op.getOperation())
            .clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// LinalgExtDialect
//===----------------------------------------------------------------------===//

void DISCLinalgExtDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<FoldTensorCastOp>(getContext());
}

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.cc.inc"