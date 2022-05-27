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

// This file defines DISC shape related operations.

#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace disc_shape {

using llvm::StringRef;

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// disc shape Dialect Constructor
//===----------------------------------------------------------------------===//

DISCShapeDialect::DISCShapeDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<DISCShapeDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.cc.inc"
      >();
  context->loadDialect<tensor::TensorDialect>();
}

namespace {

struct LinearizeOfDelinearizeOp : public OpRewritePattern<LinearizeOp> {
  using OpRewritePattern<LinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearizeOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.multiDimIndexes().size()) {
      return failure();
    }

    auto delinearizeOp = dyn_cast_or_null<DelinearizeOp>(
        op.multiDimIndexes().front().getDefiningOp());
    if (!delinearizeOp) return failure();

    if (op.multiDimIndexes().size() != delinearizeOp->getResults().size())
      return failure();

    for (auto&& z :
         llvm::zip(op.multiDimIndexes(), delinearizeOp->getResults())) {
      if (std::get<0>(z) != std::get<1>(z)) return failure();
    }

    for (auto&& z :
         llvm::zip(delinearizeOp.shapeDimIndexes(), op.shapeDimIndexes())) {
      if (std::get<0>(z) != std::get<1>(z)) return failure();
    }

    rewriter.replaceOp(op, {delinearizeOp.linearIndex()});
    return success();
  }
};

// Convert pattern like:
//   %0 = disc_shape.linearize(%i0, %i1, ..., %d0, %d1, ...)
// to:
//   %0 = disc_shape.linearize(%i0, ..., %d0, ...)
// where:
//   %d1 = constant 1
// The above simplication is motivated by following computation pattern:
//   TF computation:
//     %1 = tf.ExpandOp(%0) : (tensor<?x?xf32>) -> tensor<?x?x1xf32>
//   mhlo computation:
//     %1 = mhlo.dynamic_reshape(%0) : (tensor<?x?xf32>) -> tensor<?x?x1xf32>
// After lowering to loops:
//   scf.parallel (%i0, %i1, %i2 = (%c0, ... to %d0, %d1, %c1) step (%c1, ...) {
//     // converting to the linear index of the result value of reshape op
//     %0 = tie_shape.linearize(%i0, %i1, %i2, %d0, %d1, %c1)
//     // converting to the multi-dim indexes of the operand value of reshape op
//     %t0, %t1 = tie_shape.delinearize(%0, %d0, %d1)
//     use(%t0, %t1) // e.g. lmhlo.transpose(...)
//     ...
//   }
// After apply this canonicalization pattern:
//   scf.parallel (%i0, %i1, %i2 = (%c0, ... to %d0, %d1, %c1) step (%c1, ...) {
//     use(%i0, %i1) // e.g. lmhlo.transpose(...)
//     ...
//   }
struct RemoveSizeOneDimOfLinearizeOp : public OpRewritePattern<LinearizeOp> {
  using OpRewritePattern<LinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearizeOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.multiDimIndexes().size()) {
      return failure();
    }

    SmallVector<Value> newMultiDimIndexes;
    SmallVector<Value> newShapeDimIndexes;
    for (auto&& z : llvm::zip(op.multiDimIndexes(), op.shapeDimIndexes())) {
      Value idx = std::get<0>(z);
      Value dimSize = std::get<1>(z);
      auto constOp =
          dyn_cast_or_null<arith::ConstantIndexOp>(dimSize.getDefiningOp());
      if (constOp && constOp.getValue().cast<IntegerAttr>().getInt() == 1)
        continue;
      newMultiDimIndexes.push_back(idx);
      newShapeDimIndexes.push_back(dimSize);
    }

    if (newMultiDimIndexes.size() == op.multiDimIndexes().size()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<LinearizeOp>(
        op, op.getType(), newMultiDimIndexes, newShapeDimIndexes);
    return success();
  }
};

struct DelinearizeOfLinearizeOp : public OpRewritePattern<DelinearizeOp> {
  using OpRewritePattern<DelinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DelinearizeOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.shapeDimIndexes().size()) {
      return failure();
    }

    auto linearizeOp =
        dyn_cast_or_null<LinearizeOp>(op.linearIndex().getDefiningOp());
    if (!linearizeOp) return failure();

    if (linearizeOp.shapeDimIndexes().size() != op.shapeDimIndexes().size())
      return failure();

    for (auto&& z :
         llvm::zip(linearizeOp.shapeDimIndexes(), op.shapeDimIndexes())) {
      if (std::get<0>(z) != std::get<1>(z)) return failure();
    }

    rewriter.replaceOp(op, linearizeOp.multiDimIndexes());
    return success();
  }
};

// Convert pattern like:
//   %i0, %i1, ... = disc_shape.delinearize(%linear, %d0, %d1, ...)
// to:
//   %i0, %zero, ... = disc_shape.delinearize(%linear, %d0, ...)
// where:
//   %d1 = constant 1
struct RemoveSizeOneDimOfDelinearizeOp
    : public OpRewritePattern<DelinearizeOp> {
  using OpRewritePattern<DelinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DelinearizeOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.shapeDimIndexes().size()) {
      return failure();
    }

    SmallVector<bool> sizeOneDimVec;
    SmallVector<Value> newShapeDimIndexes;
    SmallVector<Type> newResultTypes;
    for (Value dimSize : op.shapeDimIndexes()) {
      auto constOp =
          dyn_cast_or_null<arith::ConstantIndexOp>(dimSize.getDefiningOp());
      bool isSizeOne =
          (constOp && constOp.getValue().cast<IntegerAttr>().getInt() == 1);
      sizeOneDimVec.push_back(isSizeOne);
      if (!isSizeOne) {
        newShapeDimIndexes.push_back(dimSize);
        newResultTypes.push_back(rewriter.getIndexType());
      }
    }

    if (newShapeDimIndexes.size() == op.shapeDimIndexes().size()) {
      return failure();
    }

    auto newOp = rewriter.create<DelinearizeOp>(
        op.getLoc(), newResultTypes, op.linearIndex(), newShapeDimIndexes);
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    int nonSizeOneDimIdx = 0;
    SmallVector<Value> newResults;
    for (bool isSizeOne : sizeOneDimVec) {
      if (isSizeOne) {
        newResults.push_back(zero);
      } else {
        newResults.push_back(newOp->getResult(nonSizeOneDimIdx++));
      }
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct IdentityTieShapeOp : public OpRewritePattern<TieShapeOp> {
  using OpRewritePattern<TieShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TieShapeOp op,
                                PatternRewriter& rewriter) const override {
    // Do not touch tie_shape op with symbolic dim ref attrs.
    if (op->hasAttr(SymbolicDimOp::getSymbolicDimAttrName())) return success();
    Value operand = op.value();
    auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandTy) return failure();

    // disc_shape.tie_shape(%0, %d0, %d1, ...), where
    //   %d0 = tensor.dim %0, %c0, or %d0 is a constant
    //   %d1 = tensor.dim %0, %c1, or %d1 is a constant
    bool allDimMatch = true;
    for (auto& en : llvm::enumerate(
             llvm::zip(operandTy.getShape(), op.shapeDimIndexes()))) {
      int64_t idx = en.index();
      int64_t staticDim = std::get<0>(en.value());
      Value dynamicDim = std::get<1>(en.value());
      // Skip static known dimension.
      if (staticDim != ShapedType::kDynamicSize) continue;

      auto dimOp = dyn_cast_or_null<tensor::DimOp>(dynamicDim.getDefiningOp());
      if (!dimOp || dimOp.source() != operand) {
        allDimMatch = false;
        break;
      }
      auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(
          dimOp.index().getDefiningOp());
      if (!indexOp || indexOp.getValue().cast<IntegerAttr>().getInt() != idx) {
        allDimMatch = false;
        break;
      }
    }
    if (!allDimMatch) return failure();

    rewriter.replaceOp(op, {operand});
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// LinearizeOp
//===----------------------------------------------------------------------===//

void LinearizeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  // clang-format off
  results.insert<
    LinearizeOfDelinearizeOp,
    RemoveSizeOneDimOfLinearizeOp
  >(context);
  // clang-format on
}

LogicalResult LinearizeOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// DelinearizeOp
//===----------------------------------------------------------------------===//

void DelinearizeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  // clang-format off
  results.insert<
    DelinearizeOfLinearizeOp,
    RemoveSizeOneDimOfDelinearizeOp
  >(context);
  // clang-format on
}

LogicalResult DelinearizeOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// TieShapeOp
//===----------------------------------------------------------------------===//

void TieShapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.insert<IdentityTieShapeOp>(context);
}

LogicalResult TieShapeOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// SymbolicDimOp
//===----------------------------------------------------------------------===//

void SymbolicDimOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {}

LogicalResult SymbolicDimOp::verify() { return Verify(*this); }

int64_t SymbolicDimOp::getDimSize() {
  if (auto attr = (*this)->getAttrOfType<IntegerAttr>("value"))
    return attr.getInt();
  return ShapedType::kDynamicSize;
}

void SymbolicDimOp::setDimSize(int64_t val) {
  OpBuilder b(*this);
  (*this)->setAttr("value", b.getI64IntegerAttr(val));
  if (val == -1) {
    setKnownNegativeOne(true);
  } else if (val >= 0) {
    setKnownNonNegative(true);
    if (val != 0) setKnownNonSizeZero(true);
    if (val != 1) setKnownNonSizeOne(true);
  }
}

bool SymbolicDimOp::isDynamic() {
  return getDimSize() == ShapedType::kDynamicSize;
}

void SymbolicDimOp::setKnownNonNegative(bool flag) {
  OpBuilder b(*this);
  (*this)->setAttr("knownNonNegative", b.getBoolAttr(flag));
}

void SymbolicDimOp::setKnownNegativeOne(bool flag) {
  OpBuilder b(*this);
  (*this)->setAttr("knownNegativeOne", b.getBoolAttr(flag));
  if (flag) {
    setKnownNonSizeOne(true);
    setKnownNonSizeZero(true);
  }
}

void SymbolicDimOp::setKnownNonSizeOne(bool flag) {
  OpBuilder b(*this);
  (*this)->setAttr("knownNonSizeOne", b.getBoolAttr(flag));
}

void SymbolicDimOp::setKnownNonSizeZero(bool flag) {
  OpBuilder b(*this);
  (*this)->setAttr("knownNonSizeZero", b.getBoolAttr(flag));
}

LogicalResult SymbolicDimOp::Merge(SymbolicDimOp other) {
  if (!isDynamic() && !other.isDynamic() && getDimSize() != other.getDimSize())
    return failure();
  if (isDynamic() && !other.isDynamic()) setDimSize(other.getDimSize());

  bool knownNonNegativeFlag = knownNonNegative() || other.knownNonNegative();
  bool knownNegativeOneFlag = knownNegativeOne() || other.knownNegativeOne();
  bool knownNonSizeOneFlag =
      knownNonSizeOne() || other.knownNonSizeOne() || knownNegativeOneFlag;
  bool knownNonSizeZeroFlag =
      knownNonSizeZero() || other.knownNonSizeZero() || knownNegativeOneFlag;

  if (knownNonNegativeFlag && knownNegativeOneFlag) return failure();

  setKnownNonSizeZero(knownNonSizeZeroFlag);
  setKnownNonSizeOne(knownNonSizeOneFlag);
  setKnownNegativeOne(knownNegativeOneFlag);
  setKnownNonNegative(knownNonNegativeFlag);

  return success();
}

}  // namespace disc_shape
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.cc.inc"
