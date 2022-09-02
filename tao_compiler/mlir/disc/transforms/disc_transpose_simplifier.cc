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

// This file implements the logic to remove some redundant transpose ops.
// The basic flow is shown as below:
//     input -> transpose(1, 0) -> add -> bcast -> transpose(1, 0) -> output
// is optimized to:
//     input -> add -> bcast -> output

#include <queue>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"            // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

#define DEBUG_TYPE "disc-transpose-simplifier"

namespace mlir {
namespace disc_ral {
namespace {

// We call `x transpose-dominate y` when following requirements holds.
// 1, x is the producer of y, that is x -> ... -> y
//
// 2, intermediate ops between x and y are all transpose-insensitive.
//    Here transpose-insensitive means that the performance of the
//    transposed version and the normal version have no major difference.
//    Examples are elemwise ops.
//
// 3, one-value-in (x) and one-value-out (y).
//    Each intermediate op can only have oeprands from other intermediate
//    ops, x, or some transpose-invariant (e.g. const) ops .
//    Each intermediate op can only be consumered by other intermediate or
//    y without taking shape-consumers (e.g. DimOp) into consideration.

// If x transpose-dominate y holds, we have some very nice properties.
// For example:
//    x -> ... -> y -> transpose // v0
//        can be transformed into
//    transpose -> x' -> ... -> y' // v1
//
//    v0 and v1 are supposed to have simliar performance due to the
//    following reasons:
//      ** No more transpose ops are introduced. **
//      ** intermediate ops are all transpose-insensitive. **
//
// transpose-dominate concept is very useful for transpose optimization.
// For example:
//    transpose -> x -> ... -> y -> transpsoe^{-1} // v0
//      <-> // performace is similar since.
//    transpose -> transpsoe^{-1} -> x' -> ... -> y' // v1
//      <->
//    x' -> ... -> y' // v2
// The above transform is more performant by design:
//    v0 ~ v1 < v2

void addCanonicalizationPatterns(MLIRContext* context,
                                 RewritePatternSet* patterns) {
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

template <typename T>
bool isIdentityPermutation(T&& permutation) {
  for (const auto& en : llvm::enumerate(permutation)) {
    if (en.value() != en.index()) return false;
  }
  return true;
}

// convert:
//   %1 = transpose(%0) {permutation=[0, 1, 2]} // identity transpose
//   use(%1)
// to:
//   use(%0)
LogicalResult eliminateIdentityTranspse(mhlo::TransposeOp op,
                                        PatternRewriter& rewriter) {
  if (!isIdentityPermutation(op.permutation().getValues<int64_t>()))
    return failure();
  rewriter.replaceOp(op, op->getOperands());
  return success();
}

// convert:
//   %1 = transpose(%0) {permutation=[0, 2, 1]}
//   %2 = tensor.dim %1, %c2
// to:
//   %2 = tensor.dim %0, %c1
LogicalResult propagateDimOfTranspse(tensor::DimOp op,
                                     PatternRewriter& rewriter) {
  auto indexOp =
      dyn_cast_or_null<arith::ConstantIndexOp>(op.getIndex().getDefiningOp());
  auto transposeOp =
      dyn_cast_or_null<mhlo::TransposeOp>(op.getSource().getDefiningOp());
  if (!indexOp || !transposeOp) return failure();

  auto perm = transposeOp.permutation().getValues<int64_t>();
  Value sourceIndex = rewriter.create<arith::ConstantIndexOp>(
      op.getLoc(), perm[indexOp.getValue().cast<IntegerAttr>().getInt()]);
  rewriter.replaceOpWithNewOp<tensor::DimOp>(op, transposeOp->getOperand(0),
                                             sourceIndex);
  return success();
}

// convert:
//   %1 = transpose(%0) {permutation=[0, 2, 1]}
//   %2 = shape.shape_of %1
// to:
//   %1 = transpose(%0) {permutation=[0, 2, 1]}
//   %d0 = tensor.dim %1, %c0
//   %d1 = tensor.dim %1, %c1
//   %d2 = tensor.dim %1, %c2
//   %2 = tensor.from_elements %d0, %d1, %d2
LogicalResult propagateShapeOfTranspse(shape::ShapeOfOp op,
                                       PatternRewriter& rewriter) {
  auto resultTy = op.getResult().getType().dyn_cast<RankedTensorType>();
  auto type = op.getArg().getType().dyn_cast<RankedTensorType>();
  auto transposeOp =
      dyn_cast_or_null<mhlo::TransposeOp>(op.getArg().getDefiningOp());
  if (!resultTy || !resultTy.hasStaticShape() ||
      !resultTy.getElementType().isIndex() || !type || !transposeOp)
    return failure();

  int rank = type.getRank();
  SmallVector<Value> dimValues;
  for (int i = 0; i < rank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
    dimValues.push_back(
        rewriter.create<tensor::DimOp>(op.getLoc(), op.getArg(), idx));
  }
  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, dimValues);
  return success();
}

// convert:
//   %0 = mhlo.constant dense<1> : tensor<f32>
//   %1 = bcast %0, ... : tensor<?x?xf32>
//   %2 = mhlo.mul %1, %arg : tensor<?x?xf32>
//   %use(%2)
// to:
//   %use(%arg)
bool isConstOneBcast(Operation* op) {
  if (!op) return false;
  if (!isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp,
           mhlo::DynamicBroadcastInDimOp>(op))
    return false;
  auto constOp =
      dyn_cast_or_null<mhlo::ConstantOp>(op->getOperand(0).getDefiningOp());
  if (!constOp) return false;
  auto attr = constOp.value().cast<DenseElementsAttr>();
  if (attr.getNumElements() != 1) return false;
  if (getElementTypeOrSelf(attr.getType()).isa<FloatType>())
    return (*(attr.getValues<APFloat>().begin())).convertToDouble() == 1.0;
  if (getElementTypeOrSelf(attr.getType()).isa<IntegerType>())
    return (*(attr.getValues<APInt>().begin())).isOne();
  return false;
}

LogicalResult propagateIdentityMulOp(mhlo::MulOp op,
                                     PatternRewriter& rewriter) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  auto lhsDefiningOp = lhs.getDefiningOp();
  auto rhsDefiningOp = rhs.getDefiningOp();

  if (isConstOneBcast(lhsDefiningOp)) {
    rewriter.replaceOp(op, {rhs});
    return success();
  }
  if (isConstOneBcast(rhsDefiningOp)) {
    rewriter.replaceOp(op, {lhs});
    return success();
  }
  return failure();
}

void populateTransposeSimplifierPatterns(RewritePatternSet& patterns) {
  patterns.insert(eliminateIdentityTranspse);
  patterns.insert(propagateDimOfTranspse);
  patterns.insert(propagateShapeOfTranspse);
  patterns.insert(propagateIdentityMulOp);
  // Adds canonicalization patterns to the list of patterns.
  addCanonicalizationPatterns(patterns.getContext(), &patterns);
}

SmallVector<int64_t> getReversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> newPerm(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    newPerm[perm[i]] = i;
  }
  return newPerm;
}

// Returns true is x and y are transpsoe ops and `y(x(INPUT)) = INPUT`.
bool isMirroredTranspose(Operation* x, Operation* y) {
  auto transposeX = dyn_cast_or_null<mhlo::TransposeOp>(x);
  auto transposeY = dyn_cast_or_null<mhlo::TransposeOp>(y);
  if (!transposeX || !transposeY) return false;

  auto permXAttr = transposeX.permutation().getValues<int64_t>();
  SmallVector<int64_t> permX{permXAttr.begin(), permXAttr.end()};
  auto permYAttr = transposeY.permutation().getValues<int64_t>();
  SmallVector<int64_t> permY{permYAttr.begin(), permYAttr.end()};

  if (permX.size() != permY.size()) return false;

  auto reverseX = getReversePermutation(permX);
  return reverseX == permY;
}

RankedTensorType getTransposeOutputType(
    Value value, const SmallVectorImpl<int64_t>& permutation, OpBuilder& b) {
  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 4> transposed_shape;
  auto input_type = value.getType().cast<RankedTensorType>();
  auto input_shape = input_type.getShape();
  for (int64_t val : permutation) {
    transposed_shape.push_back(input_shape[val]);
  }
  if (auto attrs = input_type.getEncoding().dyn_cast_or_null<ArrayAttr>()) {
    SmallVector<Attribute> newAttrs;
    for (int64_t val : permutation) {
      newAttrs.push_back(attrs[val]);
    }
    auto symbolicShapeAttr = ArrayAttr::get(value.getContext(), newAttrs);
    return RankedTensorType::get(transposed_shape, input_type.getElementType(),
                                 symbolicShapeAttr);
  }
  return RankedTensorType::get(transposed_shape, input_type.getElementType());
}

Operation* insertTranspose(Operation* op, Value value,
                           const SmallVectorImpl<int64_t>& permutation,
                           OpBuilder& b) {
  auto permutationAttr = GetI64ElementsAttr(permutation, &b);

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 4> transposed_shape;
  ShapedType input_type = value.getType().cast<ShapedType>();
  auto input_shape = input_type.getShape();
  for (auto val : permutation) {
    transposed_shape.push_back(input_shape[val]);
  }
  auto transpose_type = getTransposeOutputType(value, permutation, b);
  auto transpose_op = b.create<mhlo::TransposeOp>(op->getLoc(), transpose_type,
                                                  value, permutationAttr);

  if (auto attr = op->getAttr(placement_utils::kDiscPlaceAssignment))
    transpose_op->setAttr(placement_utils::kDiscPlaceAssignment, attr);

  return transpose_op;
}

// (value, permutation)
using ValueTransposeRequest = std::pair<Value, SmallVector<int64_t>>;

struct TransposeSimpliferContext {
  explicit TransposeSimpliferContext(Block* block);

  bool dominates(Value from, Value to, const SmallVector<int64_t>& permutation,
                 DenseMap<Operation*, SmallVector<int64_t>>& permMap);

  LogicalResult rewriteIntermediateOps(
      DenseMap<Operation*, SmallVector<int64_t>>& permMap);

  bool inTargetBlock(Operation* op) {
    return llvm::find(opList, op) != opList.end();
  }

  // topological order. Producer comes first.
  SmallVector<Operation*> opList;

  // Map an operation in the block to an index that represents its order
  // in a topological sequence. Producer has lower index.
  DenseMap<Operation*, int> op2Idx;
};

TransposeSimpliferContext::TransposeSimpliferContext(Block* block) {
  for (Operation& op : *block) {
    op2Idx[&op] = opList.size();
    opList.push_back(&op);
  }
}

LogicalResult backwardBcastPermutation(Operation* op, ArrayRef<int64_t> perm,
                                       SmallVector<int64_t>& operandPerm,
                                       SmallVector<int64_t>& bcastDims) {
  Value operand = op->getOperand(0);
  auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
  assert(dimAttr);
  auto dimensions = dimAttr.getValues<int64_t>();
  DenseMap<int64_t, int64_t> out2InMapping;
  for (const auto& en : llvm::enumerate(dimensions))
    out2InMapping[en.value()] = en.index();
  for (const auto& en : llvm::enumerate(perm)) {
    if (llvm::find(dimensions, en.value()) == dimensions.end()) continue;
    bcastDims.push_back(en.index());
    operandPerm.push_back(out2InMapping[en.value()]);
  }

  return success();
}

LogicalResult backwardPermutation(Operation* op, int operandIdx,
                                  const SmallVector<int64_t>& perm,
                                  SmallVector<int64_t>& operandPerm) {
  if (op->getDialect() != op->getContext()->getLoadedDialect("mhlo") &&
      op->getDialect() != op->getContext()->getLoadedDialect("mhlo_disc")) {
    return failure();
  }

  operandPerm.clear();
  Value operand = op->getOperand(operandIdx);
  if (op->hasTrait<mlir::OpTrait::Elementwise>() ||
      dyn_cast<mhlo::ClampOp>(op)) {
    auto type = operand.getType().dyn_cast<RankedTensorType>();
    if (!type) return failure();
    if (type.getRank() > 0) {
      operandPerm = perm;
    }
    return success();
  } else if (isa<mhlo::BroadcastInDimOp, mhlo::DynamicBroadcastInDimOp>(op)) {
    // shape operand does not need transpose.
    if (operandIdx > 0) return success();
    SmallVector<int64_t> bcastDims;
    return backwardBcastPermutation(op, perm, operandPerm, bcastDims);
  }

  // unknown ops
  return failure();
}

bool TransposeSimpliferContext::dominates(
    Value from, Value to, const SmallVector<int64_t>& permutation,
    DenseMap<Operation*, SmallVector<int64_t>>& permMap) {
  LLVM_DEBUG(llvm::dbgs() << "transpose dominant:\n\tfrom: " << from
                          << "\n\tto:" << to << "\n");
  Operation* out = nullptr;
  for (Operation* user : to.getUsers()) {
    if (isa<tensor::DimOp, shape::ShapeOfOp>(user)) continue;
    // multiple consumers.
    if (out) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "transpose dominant: multiple outside consumers detected\n\t"
          << *out << "\n\t" << *user << "\n");
      return false;
    }
    out = user;
  }
  if (!inTargetBlock(out)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "transpose dominant: consumers is not in the same block, consumer = "
        << out);
    if (out) {
      LLVM_DEBUG(llvm::dbgs() << "\tconsumer:  " << *out << "\n");
    }
    return false;
  }
  SmallVector<ValueTransposeRequest> queue{{to, permutation}};
  DenseMap<Value, std::set<SmallVector<int64_t>>> visitedValues;
  DenseSet<Operation*> consumers;
  DenseSet<Operation*> intermidateOps{out};

  auto tryEnqueue = [&](Operation* op, const SmallVector<int64_t>& perm) {
    for (const auto& en : llvm::enumerate(op->getOperands())) {
      SmallVector<int64_t> operandPerm;
      if (failed(backwardPermutation(op, en.index(), perm, operandPerm))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "transpose dominant: backward permutation failed for "
                   << *op << " operand #" << en.index() << "\n");
        return false;
      }
      if (isIdentityPermutation(operandPerm)) continue;
      auto it = visitedValues[en.value()].insert(operandPerm);
      if (!it.second) continue;
      // inconsistent transpose request for the same value.
      if (visitedValues[en.value()].size() > 1) {
        LLVM_DEBUG(llvm::dbgs() << "transpose dominant: inconsistent transpose "
                                   "request for the same value: "
                                << en.value() << "\n");
        return false;
      }
      ValueTransposeRequest request{en.value(), operandPerm};
      queue.emplace_back(std::move(request));
    }
    return true;
  };

  while (!queue.empty()) {
    Value val;
    SmallVector<int64_t> perm;
    std::tie(val, perm) = queue.back();
    queue.pop_back();

    if (val == from) continue;
    Operation* definingOp = val.getDefiningOp();
    if (!definingOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "transpose dominant: block arg is not supported\n");
      return false;
    }
    if (isa<mhlo::ConstantOp>(definingOp)) continue;

    // producer should in the same block.
    if (!inTargetBlock(definingOp)) {
      LLVM_DEBUG(llvm::dbgs() << "transpose dominant: producer is not in the "
                                 "same block, producer = "
                              << *definingOp);
      return false;
    }

    auto it = permMap.find(definingOp);
    if (it != permMap.end() && it->second != perm) {
      LLVM_DEBUG(llvm::dbgs() << "transpose dominant: inconsistent transpose "
                                 "request for the same op: "
                              << *definingOp << "\n");
      return false;
    }
    permMap[definingOp] = perm;

    if (!tryEnqueue(definingOp, perm)) return false;
    for (Operation* user : val.getUsers()) {
      if (isa<tensor::DimOp, shape::ShapeOfOp>(user)) continue;
      consumers.insert(user);
    }
    intermidateOps.insert(definingOp);
  }
  for (Operation* consumer : consumers)
    if (!intermidateOps.count(consumer)) {
      LLVM_DEBUG(llvm::dbgs() << "transpose dominant: detect outside consumer "
                                 "not in intermidate ops set: "
                              << *consumer << "\n");
      return false;
    }
  return true;
}

LogicalResult TransposeSimpliferContext::rewriteIntermediateOps(
    DenseMap<Operation*, SmallVector<int64_t>>& permMap) {
  for (auto& pair : permMap) {
    Operation* op;
    SmallVector<int64_t> perm;
    std::tie(op, perm) = pair;
    SmallVector<int64_t> reversePerm = getReversePermutation(perm);
    OpBuilder b(op);
    Location loc = op->getLoc();

    auto pushRefinedOperand = [&](int operandIdx,
                                  SmallVector<Value>& newOperands) {
      Value operand = op->getOperand(operandIdx);
      SmallVector<int64_t> operandPerm;
      if (failed(backwardPermutation(op, operandIdx, perm, operandPerm)))
        return false;
      if (isIdentityPermutation(operandPerm)) {
        newOperands.push_back(operand);
      } else {
        newOperands.push_back(
            insertTranspose(op, operand, perm, b)->getResult(0));
      }
      return true;
    };

    auto cloneOpWithNewOperands = [&](SmallVector<Value>& newOperands) {
      Operation* clonedOp = b.clone(*op);
      clonedOp->setOperands(newOperands);
      SmallVector<Value> newResults = clonedOp->getResults();
      for (const auto& en : llvm::enumerate(op->getResults())) {
        Type newType = getTransposeOutputType(en.value(), perm, b);
        newResults[en.index()].setType(newType);
      }
      return clonedOp;
    };

    auto replaceOpWith = [&](Operation* newOp) {
      for (const auto& en : llvm::zip(newOp->getResults(), op->getResults())) {
        Value newValue = std::get<0>(en);
        Value oldValue = std::get<1>(en);
        Value newResult =
            insertTranspose(op, newValue, reversePerm, b)->getResult(0);
        oldValue.replaceAllUsesWith(newResult);
      }
    };

    if (op->hasTrait<mlir::OpTrait::Elementwise>() ||
        dyn_cast<mhlo::ClampOp>(op)) {
      SmallVector<Value> newOperands;
      for (int i = 0; i < op->getNumOperands(); ++i)
        if (!pushRefinedOperand(i, newOperands)) return failure();
      Operation* clonedOp = cloneOpWithNewOperands(newOperands);
      replaceOpWith(clonedOp);
    } else if (isa<mhlo::BroadcastInDimOp, mhlo::DynamicBroadcastInDimOp>(op)) {
      SmallVector<Value> newOperands;
      if (!pushRefinedOperand(0, newOperands)) return failure();
      if (isa<mhlo::DynamicBroadcastInDimOp>(op)) {
        Value shapeOperand = op->getOperand(1);
        int rank =
            shapeOperand.getType().cast<RankedTensorType>().getDimSize(0);
        assert(rank > 0);
        SmallVector<Value> oldShapeValues;
        for (int i = 0; i < rank; ++i) {
          Value idx = b.create<arith::ConstantIndexOp>(loc, i);
          oldShapeValues.push_back(
              b.create<tensor::ExtractOp>(loc, shapeOperand, idx));
        }
        SmallVector<Value> newShapeValues(rank);
        for (int i = 0; i < rank; ++i)
          newShapeValues[i] = oldShapeValues[perm[i]];
        Value newShapeOperand =
            b.create<tensor::FromElementsOp>(loc, newShapeValues);
        newOperands.push_back(newShapeOperand);
      }
      Operation* clonedOp = cloneOpWithNewOperands(newOperands);
      SmallVector<int64_t> operandPerm, bcastDims;
      if (failed(backwardBcastPermutation(op, perm, operandPerm, bcastDims)))
        return failure();
      clonedOp->setAttr("broadcast_dimensions",
                        GetI64ElementsAttr(bcastDims, &b));
      replaceOpWith(clonedOp);
    } else {
      // unknown ops
      return failure();
    }
  }

  return success();
}

// convert:
//   transpose(1, 0) -> x -> ... -> y -> transpsoe(1, 0)
//     ->
//   transpose(1, 0) -> transpsoe(1, 0) -> x' -> ... -> y'
LogicalResult pairMirroredTransposeOps(Block* block, bool& changed) {
  SmallVector<Operation*> transposeOps;
  for (Operation& op : *block) {
    if (isa<mhlo::TransposeOp>(&op)) transposeOps.push_back(&op);
  }
  // Early stop if no enough candidate transpose ops.
  int numTransposeOps = transposeOps.size();
  if (numTransposeOps < 2) return success();

  TransposeSimpliferContext ctx(block);
  for (int i = 0; i < numTransposeOps; ++i) {
    for (int j = i + 1; j < numTransposeOps; ++j) {
      Operation* x = transposeOps[i];
      Operation* y = transposeOps[j];
      LLVM_DEBUG(llvm::dbgs() << "Try transpose dominant\n");
      LLVM_DEBUG(llvm::dbgs() << "\tfrom: " << *x << "\n");
      LLVM_DEBUG(llvm::dbgs() << "\tto  : " << *y << "\n");
      if (!isMirroredTranspose(x, y)) continue;
      LLVM_DEBUG(llvm::dbgs() << "\tmirrored transpose check passed\n");
      Value from = x->getResult(0);
      Value to = y->getOperand(0);
      auto permAttr =
          cast<mhlo::TransposeOp>(y).permutation().getValues<int64_t>();
      SmallVector<int64_t> perm{permAttr.begin(), permAttr.end()};
      DenseMap<Operation*, SmallVector<int64_t>> intermedateOpsPermutationMap;
      if (!ctx.dominates(from, to, perm, intermedateOpsPermutationMap))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "\tdominant check passed\n");
      if (failed(ctx.rewriteIntermediateOps(intermedateOpsPermutationMap)))
        return x->emitError("failed to rewrite intermidate ops");
      changed = true;
      return success();
    }
  }
  return success();
}

bool checkIsOnlyNonShapeUser(Value val, Operation* op) {
  Operation* prev = nullptr;
  for (Operation* user : val.getUsers()) {
    if (isa<tensor::DimOp, shape::ShapeOfOp>(user)) continue;
    if (user != op) return false;
    prev = user;
  }
  return prev != nullptr;
}

mhlo::TransposeOp findTransposeProducer(Value val, Operation* op) {
  auto definingOp = val.getDefiningOp();
  // only do the optimization inside a basic block.
  if (!definingOp || definingOp->getBlock() != op->getBlock()) return nullptr;
  if (!checkIsOnlyNonShapeUser(val, op)) return nullptr;

  auto transposeOp = dyn_cast<mhlo::TransposeOp>(definingOp);
  if (transposeOp) return transposeOp;

  if (definingOp->getDialect() !=
          definingOp->getContext()->getLoadedDialect("mhlo") &&
      definingOp->getDialect() !=
          definingOp->getContext()->getLoadedDialect("mhlo_disc")) {
    return nullptr;
  }

  if (isa<mhlo::BroadcastInDimOp, mhlo::DynamicBroadcastInDimOp>(definingOp)) {
    return findTransposeProducer(definingOp->getOperand(0), definingOp);
  }

  if (!definingOp->hasTrait<mlir::OpTrait::Elementwise>()) return nullptr;

  if (definingOp->getNumOperands() == 1) {
    return findTransposeProducer(definingOp->getOperand(0), definingOp);
  } else if (definingOp->getNumOperands() == 2) {
    auto lhsDefiningOp = definingOp->getOperand(0).getDefiningOp();
    auto rhsDefiningOp = definingOp->getOperand(1).getDefiningOp();
    if (dyn_cast_or_null<mhlo::ConstantOp>(lhsDefiningOp))
      return findTransposeProducer(definingOp->getOperand(1), definingOp);
    if (dyn_cast_or_null<mhlo::ConstantOp>(rhsDefiningOp))
      return findTransposeProducer(definingOp->getOperand(0), definingOp);
    // lhs: bcast + const
    if (dyn_cast_or_null<mhlo::BroadcastInDimOp>(lhsDefiningOp) ||
        dyn_cast_or_null<mhlo::DynamicBroadcastInDimOp>(lhsDefiningOp)) {
      auto prevOp = lhsDefiningOp->getOperand(0).getDefiningOp();
      if (dyn_cast_or_null<mhlo::ConstantOp>(prevOp))
        return findTransposeProducer(definingOp->getOperand(1), definingOp);
    }
    // rhs: bcast + const
    if (dyn_cast_or_null<mhlo::BroadcastInDimOp>(rhsDefiningOp) ||
        dyn_cast_or_null<mhlo::DynamicBroadcastInDimOp>(rhsDefiningOp)) {
      auto prevOp = rhsDefiningOp->getOperand(0).getDefiningOp();
      if (dyn_cast_or_null<mhlo::ConstantOp>(prevOp))
        return findTransposeProducer(definingOp->getOperand(0), definingOp);
    }
    return nullptr;
  }
  return nullptr;
}

// Basic idea:
//  convert:
//   x -> transpose ---
//                      \
//                       v
//   y -> transpose --> add --> ...
//  to:
//   x ---
//        \
//         v
//   y --> add -> transpose -> ...
//
LogicalResult reverseIfOperandsAreConsistentTransposeOps(Operation* op,
                                                         bool& changed) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  auto transposeLHS = findTransposeProducer(lhs, op);
  auto transposeRHS = findTransposeProducer(rhs, op);
  if (!transposeLHS || !transposeRHS || transposeLHS == transposeRHS)
    return success();

  SmallVector<int64_t> permLHS{transposeLHS.permutation().getValues<int64_t>()};
  SmallVector<int64_t> permRHS{transposeRHS.permutation().getValues<int64_t>()};
  if (permLHS.size() != permRHS.size()) return success();

  for (const auto& en : llvm::zip(permLHS, permRHS)) {
    if (std::get<0>(en) != std::get<1>(en)) return success();
  }

  OpBuilder b(op);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertTranspose(op, lhs, permLHS, b)->getResult(0));
  newOperands.push_back(insertTranspose(op, rhs, permLHS, b)->getResult(0));
  Operation* clonedOp = b.clone(*op);
  clonedOp->setOperands(newOperands);
  Type newType = getTransposeOutputType(op->getResult(0), permLHS, b);
  clonedOp->getResult(0).setType(newType);
  SmallVector<int64_t> reversePerm = getReversePermutation(permLHS);
  Value newResult =
      insertTranspose(op, clonedOp->getResult(0), reversePerm, b)->getResult(0);
  op->getResult(0).replaceAllUsesWith(newResult);

  changed = true;
  return success();
}

// Basic idea:
//  convert:
//                      -> ... -> transpose -> xxx
//                     /
//                    x ---
//                          \
//                           v
//   y -> transpose^{-1} -> add -> ... -> transpose -> yyy
//                              \
//                               -> ... -> zzz
//  to:
//                        -> ... -> xxx
//                       /
//     x -> transpose(1, 0)
//                          \
//                           v
//                     y -> add -> ... -> yyy
//                              \
//                               -> transpose^{-1} ... -> zzz
//  or convert:
//                const ---
//                          \
//                           v
//   y -> transpose^{-1} -> add -> ... -> transpose -> yyy
//                              \
//                               -> ... -> zzz
//  to:
//                const' --
//                          \
//                           v
//                     y -> add -> ... -> yyy
//                              \
//                               -> transpose^{-1} ... -> zzz
LogicalResult reverseIfOperandsAndResultsAreConsistent(Operation* op,
                                                       bool& changed) {
  LLVM_DEBUG(llvm::dbgs() << "reverseIfOperandsAndResultsAreConsistent: " << *op
                          << "\n");
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  auto transposeLHS = findTransposeProducer(lhs, op);
  auto transposeRHS = findTransposeProducer(rhs, op);
  if ((transposeLHS == nullptr) == (transposeRHS == nullptr)) return success();

  if (transposeLHS) {
    LLVM_DEBUG(llvm::dbgs() << "\ttransposeLHS: " << transposeLHS << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "\ttransposeRHS: " << transposeRHS << "\n");
  }

  mhlo::TransposeOp transposeOp = transposeLHS ? transposeLHS : transposeRHS;
  Value otherVal = transposeLHS ? rhs : lhs;
  Operation* otherDefiningOp = otherVal.getDefiningOp();
  // in case there is a bcast op due to implicit bcast.
  bool implicit_bcast = false;
  if (otherDefiningOp &&
      isa<mhlo::BroadcastInDimOp, mhlo::DynamicBroadcastInDimOp>(
          otherDefiningOp)) {
    if (checkIsOnlyNonShapeUser(otherVal, op)) {
      implicit_bcast = true;
      otherVal = otherDefiningOp->getOperand(0);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "\timplicit_bcast: " << implicit_bcast << "\n");
  LLVM_DEBUG(llvm::dbgs() << "\totherVal: " << otherVal << "\n");

  Block* block = op->getBlock();
  SmallVector<int64_t> reversePerm{
      transposeOp.permutation().getValues<int64_t>()};
  auto perm = getReversePermutation(reversePerm);
  SmallVector<Operation*> transposeOps;
  for (Operation& candidate : *block) {
    auto transposeCandidate = dyn_cast<mhlo::TransposeOp>(&candidate);
    if (!transposeCandidate || &candidate == transposeOp) continue;
    SmallVector<int64_t> candidatePerm{
        transposeCandidate.permutation().getValues<int64_t>()};
    if (perm != candidatePerm) continue;
    transposeOps.push_back(&candidate);
  }

  bool input_is_const =
      (dyn_cast_or_null<mhlo::ConstantOp>(otherVal.getDefiningOp()) != nullptr);
  bool input_has_tranpose_consumer = false;
  bool output_has_tranpose_consumer = false;
  SmallVector<DenseMap<Operation*, SmallVector<int64_t>>> permMaps;
  TransposeSimpliferContext ctx(block);
  for (Operation* candidate : transposeOps) {
    if (candidate->getOperand(0) == otherVal) {
      input_has_tranpose_consumer = true;
      continue;
    }
    if (candidate->getOperand(0) == op->getResult(0)) {
      output_has_tranpose_consumer = true;
      continue;
    }

    DenseMap<Operation*, SmallVector<int64_t>> intermedateOpsPermutationMap;
    if (ctx.dominates(otherVal, candidate->getOperand(0), perm,
                      intermedateOpsPermutationMap)) {
      input_has_tranpose_consumer = true;
      permMaps.push_back(intermedateOpsPermutationMap);
    }
    intermedateOpsPermutationMap.clear();
    if (ctx.dominates(op->getResult(0), candidate->getOperand(0), perm,
                      intermedateOpsPermutationMap)) {
      output_has_tranpose_consumer = true;
      permMaps.push_back(intermedateOpsPermutationMap);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\tinput_is_const: " << input_is_const << "\n");
  LLVM_DEBUG(llvm::dbgs() << "\tinput_has_tranpose_consumer: "
                          << input_has_tranpose_consumer << "\n");
  LLVM_DEBUG(llvm::dbgs() << "\toutput_has_tranpose_consumer: "
                          << output_has_tranpose_consumer << "\n");

  if (!input_has_tranpose_consumer && !input_is_const ||
      !output_has_tranpose_consumer)
    return success();
  for (auto& permMap : permMaps)
    if (failed(ctx.rewriteIntermediateOps(permMap))) return failure();

  OpBuilder b(op);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertTranspose(op, lhs, perm, b)->getResult(0));
  newOperands.push_back(insertTranspose(op, rhs, perm, b)->getResult(0));
  Operation* clonedOp = b.clone(*op);
  clonedOp->setOperands(newOperands);
  Type newType = getTransposeOutputType(op->getResult(0), perm, b);
  clonedOp->getResult(0).setType(newType);
  Value newResult =
      insertTranspose(op, clonedOp->getResult(0), reversePerm, b)->getResult(0);
  op->getResult(0).replaceAllUsesWith(newResult);

  // convert:
  //   x -> bcast -> tranpose
  // to:
  //   x -> transpose -> bcast
  // in case having implicit bcast
  if (implicit_bcast) {
    Value newOperand = transposeLHS ? newOperands[1] : newOperands[0];
    Value transposeOperand = newOperand.getDefiningOp()->getOperand(0);
    TransposeSimpliferContext ctx(block);
    DenseMap<Operation*, SmallVector<int64_t>> intermedateOpsPermutationMap;
    bool status = ctx.dominates(otherVal, transposeOperand, perm,
                                intermedateOpsPermutationMap);
    (void)status;
    assert(status);
    if (failed(ctx.rewriteIntermediateOps(intermedateOpsPermutationMap)))
      return failure();
  }

  changed = true;
  return success();
}

LogicalResult reverseBinaryOpsIfBeneficial(Block* block, bool& changed) {
  SmallVector<Operation*> ops;
  for (Operation& op : *block) {
    if (op.getDialect() != op.getContext()->getLoadedDialect("mhlo") &&
        op.getDialect() != op.getContext()->getLoadedDialect("mhlo_disc")) {
      continue;
    }
    if (!op.hasTrait<mlir::OpTrait::Elementwise>()) continue;
    if (op.getNumOperands() != 2) continue;
    ops.push_back(&op);
  }

  for (Operation* op : ops) {
    if (failed(reverseIfOperandsAreConsistentTransposeOps(op, changed)))
      return failure();
    if (changed) return success();
    if (failed(reverseIfOperandsAndResultsAreConsistent(op, changed)))
      return failure();
    if (changed) return success();
  }
  return success();
}

struct TransposeSimplifierPass
    : public TransposeSimplifierPassBase<TransposeSimplifierPass> {
  void runOnOperation() override {
    auto status = getOperation().walk([&](Block* block) {
      if (failed(runOnBlock(block))) return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (status.wasInterrupted()) signalPassFailure();
  }

  LogicalResult runOnBlock(Block* block);
  LogicalResult runCanonicalizer(func::FuncOp func);
};

LogicalResult TransposeSimplifierPass::runCanonicalizer(func::FuncOp func) {
  RewritePatternSet patterns(func.getContext());
  populateTransposeSimplifierPatterns(patterns);
  // ignore the not-converged error since we are in a loop.
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  OpPassManager dynamicPM("builtin.func");
  dynamicPM.addPass(createCSEPass());
  return runPipeline(dynamicPM, func);
}

LogicalResult TransposeSimplifierPass::runOnBlock(Block* block) {
  bool changed;
  do {
    changed = false;

    if (failed(runCanonicalizer(getOperation()))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to do clean up");
      return failure();
    }

    if (failed(pairMirroredTransposeOps(block, changed))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to pair mirrored transpose ops");
      return failure();
    }
    if (changed) continue;

    if (failed(reverseBinaryOpsIfBeneficial(block, changed))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to pair mirrored transpose ops");
      return failure();
    }
    if (changed) continue;

    if (failed(runCanonicalizer(getOperation()))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to do clean up");
      return failure();
    }
  } while (changed);
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTransposeSimplifierPass() {
  return std::make_unique<TransposeSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir
