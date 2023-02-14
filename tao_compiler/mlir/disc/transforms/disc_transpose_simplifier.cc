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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"            // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

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
  if (!isIdentityPermutation(op.getPermutation().getValues<int64_t>()))
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

  auto perm = transposeOp.getPermutation().getValues<int64_t>();
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
  auto attr = constOp.getValue().cast<DenseElementsAttr>();
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

  auto permXAttr = transposeX.getPermutation().getValues<int64_t>();
  SmallVector<int64_t> permX{permXAttr.begin(), permXAttr.end()};
  auto permYAttr = transposeY.getPermutation().getValues<int64_t>();
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

  ArrayRef<mhlo::TransposeOp> getTransposeOps() { return transposeOps; }

  // Returns a producer graph for ops in the block:
  //   map<op-in-the-block, direct-or-indirect-producers-set-of-this-op>
  DenseMap<Operation*, DenseSet<Value>> buildProducerGraph();

  // topological order. Producer comes first.
  SmallVector<Operation*> opList;

  // topological order. Producer comes first.
  SmallVector<mhlo::TransposeOp> transposeOps;

  // Map an operation in the block to an index that represents its order
  // in a topological sequence. Producer has lower index.
  DenseMap<Operation*, int> op2Idx;
};

TransposeSimpliferContext::TransposeSimpliferContext(Block* block) {
  for (Operation& op : *block) {
    op2Idx[&op] = opList.size();
    opList.push_back(&op);
    if (auto transposeOp = dyn_cast<mhlo::TransposeOp>(&op))
      transposeOps.push_back(transposeOp);
  }
}

// Returns a producer graph for ops in the block:
//   map<op-in-the-block, direct-or-indirect-producers-set-of-this-op>
DenseMap<Operation*, DenseSet<Value>>
TransposeSimpliferContext::buildProducerGraph() {
  DenseMap<Operation*, DenseSet<Value>> graph;
  for (auto op : opList) {
    for (Value operand : op->getOperands()) {
      graph[op].insert(operand);
      auto definingOp = operand.getDefiningOp();
      if (llvm::find(opList, definingOp) == opList.end()) continue;
      for (Value producer : graph[definingOp]) graph[op].insert(producer);
    }
  }
  return graph;
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

LogicalResult backwardTransposePermutation(
    Operation* op, ArrayRef<int64_t> perm, SmallVector<int64_t>& operandPerm,
    SmallVector<int64_t>& newPermutation) {
  auto permutationAttr = op->getAttrOfType<DenseElementsAttr>("permutation");
  assert(permutationAttr);
  auto oldPerm = llvm::to_vector<>(permutationAttr.getValues<int64_t>());
  operandPerm.resize(perm.size());
  newPermutation.resize(perm.size());
  for (int d = 0; d < perm.size(); ++d) {
    // We always assign identity permutation for operand and create
    // a new transpose op with new permutation since transpose'(transpose(x))
    // can be folded to transpose''(x)
    newPermutation[d] = oldPerm[perm[d]];
    operandPerm[d] = d;
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
  } else if (auto transposeOp = dyn_cast<mhlo::TransposeOp>(op)) {
    SmallVector<int64_t> newPerm;
    return backwardTransposePermutation(op, perm, operandPerm, newPerm);
  }

  // unknown ops
  return failure();
}

LogicalResult backwardPermutation(
    Operation* op, const SmallVector<int64_t>& perm,
    DenseMap<int, SmallVector<int64_t>>& operandPermMap) {
  operandPermMap.clear();
  for (int i = 0; i < op->getNumOperands(); ++i)
    if (failed(backwardPermutation(op, i, perm, operandPermMap[i])))
      return failure();
  return success();
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
    if (isa<mhlo::TransposeOp>(definingOp)) {
      LLVM_DEBUG(llvm::dbgs() << "transpose dominant: not support to propagate "
                                 "permutation through a transpsoe op\n");
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
            insertTranspose(op, operand, operandPerm, b)->getResult(0));
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
    } else if (isa<mhlo::TransposeOp>(op)) {
      SmallVector<int64_t> operandPerm;
      SmallVector<int64_t> newPermutation;
      if (failed(backwardTransposePermutation(op, perm, operandPerm,
                                              newPermutation)))
        return failure();
      // T12 = T1(T2) // fuse T1 and T2
      auto fusedOp = insertTranspose(op, op->getOperand(0), newPermutation, b);
      replaceOpWith(fusedOp);
    } else {
      // unknown ops
      return failure();
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

  SmallVector<int64_t> permLHS{
      transposeLHS.getPermutation().getValues<int64_t>()};
  SmallVector<int64_t> permRHS{
      transposeRHS.getPermutation().getValues<int64_t>()};
  if (permLHS.size() != permRHS.size()) return success();

  for (const auto& en : llvm::zip(permLHS, permRHS)) {
    if (std::get<0>(en) != std::get<1>(en)) return success();
  }

  OpBuilder b(op);
  SmallVector<Value> newOperands;
  SmallVector<int64_t> reversePerm = getReversePermutation(permLHS);
  newOperands.push_back(insertTranspose(op, lhs, reversePerm, b)->getResult(0));
  newOperands.push_back(insertTranspose(op, rhs, reversePerm, b)->getResult(0));
  Operation* clonedOp = b.clone(*op);
  clonedOp->setOperands(newOperands);
  Type newType = getTransposeOutputType(op->getResult(0), reversePerm, b);
  clonedOp->getResult(0).setType(newType);
  Value newResult =
      insertTranspose(op, clonedOp->getResult(0), permLHS, b)->getResult(0);
  op->getResult(0).replaceAllUsesWith(newResult);

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
  }
  return success();
}

LogicalResult tryMoveUpTransposeGreedilyAndApplyIfBeneficialImpl(
    TransposeSimpliferContext& ctx, ArrayRef<mhlo::TransposeOp> transposeOps,
    bool& changed) {
  bool consistentPerm = true;
  int numTransposeEliminated = 0;
  DenseMap<Operation*, SmallVector<int64_t>> permMap;
  DenseMap<Operation*, DenseSet<Operation*>> userMap;
  // Transpose requests for each value that is outside of block.
  DenseMap<Value, SmallVector<SmallVector<int64_t>>> outsideOperandMap;
  std::priority_queue<std::pair<int, Operation*>> queue;
  // Returns true if we need to insert a new transpose op for `producer`.
  auto tryUpdate = [&](Value value, Operation* consumer,
                       const SmallVector<int64_t>& perm) {
    auto producer = value.getDefiningOp();
    // do not propagate transpose across block a.t.m.
    if (!ctx.inTargetBlock(producer)) {
      auto& perms = outsideOperandMap[value];
      if (llvm::find(perms, perm) != perms.end()) return false;
      perms.push_back(perm);
      return true;
    }

    auto it = permMap.find(producer);
    if (it != permMap.end() && it->second == perm) {
      userMap[producer].insert(consumer);
      // skip insert a new transpose op if there has been a transpose op.
      return false;
    }
    // TODO(kevin.zwy): support multiple different permutation requirements.
    consistentPerm = !(it != permMap.end());
    permMap[producer] = perm;
    userMap[producer].insert(consumer);
    queue.emplace(std::pair<int, Operation*>{ctx.op2Idx[producer], producer});
    return true;
  };

  for (auto transposeOp : transposeOps) {
    Operation* op = transposeOp.getOperation();
    tryUpdate(
        op->getOperand(0), op,
        llvm::to_vector<>(transposeOp.getPermutation().getValues<int64_t>()));
  }

  while (!queue.empty()) {
    Operation* curOp = queue.top().second;
    queue.pop();

    // firstly remove the transpose backward request for `curOp` before the
    // `backwardPermutation` check. We'll add the request back once the check is
    // passed. Those left requests after we visit all items in the queue are
    // supposed to be applied successfully.
    auto perm = permMap[curOp];
    permMap.erase(curOp);

    LLVM_DEBUG(llvm::dbgs() << "\tcurOp: " << *curOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "\tinit numTransposeEliminated for curOp: "
                            << numTransposeEliminated << "\n");
    // TODO(kevin.zwy): support op with more than one result.
    if (curOp->getNumResults() > 1) continue;
    // check if we can move up transpose op through this op.
    DenseMap<int, SmallVector<int64_t>> operandPermMap;
    if (failed(backwardPermutation(curOp, perm, operandPermMap))) continue;
    LLVM_DEBUG(llvm::dbgs() << "\tpass backwardPermutation for curOp\n");
    permMap[curOp] = perm;

    bool needReverse = false;
    auto& transposedUsers = userMap[curOp];
    LLVM_DEBUG(llvm::dbgs() << "\ttransposedUsers list of curOp ("
                            << transposedUsers.size() << "):\n");
    for (auto& transposedUser : transposedUsers) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\t transposedUser: " << *transposedUser << "\n");
    }
    for (auto user : curOp->getResult(0).getUsers()) {
      LLVM_DEBUG(llvm::dbgs() << "\tuser of curOp: " << *user << "\n");
      if (isa<tensor::DimOp, shape::ShapeOfOp, mhlo::TransposeOp>(user))
        continue;
      if (transposedUsers.find(user) != transposedUsers.end()) continue;
      needReverse = true;
    }
    if (!needReverse) {
      ++numTransposeEliminated;
      LLVM_DEBUG(llvm::dbgs()
                 << "\t++numTransposeEliminated = " << numTransposeEliminated
                 << " due to all the users of curOp are all transposed\n");
      if (isa<mhlo::TransposeOp>(curOp)) {
        SmallVector<int64_t> operandPerm;
        SmallVector<int64_t> newPermutation;
        if (failed(backwardTransposePermutation(curOp, perm, operandPerm,
                                                newPermutation)))
          return failure();
        if (isIdentityPermutation(newPermutation)) {
          ++numTransposeEliminated;
          LLVM_DEBUG(llvm::dbgs() << "\t++numTransposeEliminated = "
                                  << numTransposeEliminated
                                  << " due to curOp is transpose op and can "
                                     "be eliminated after folding\n");
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "\tpropagate transpose to operands of curOp ("
                            << curOp->getNumOperands() << ")\n");
    for (const auto& en : llvm::enumerate(curOp->getOperands())) {
      LLVM_DEBUG(llvm::dbgs() << "\t\tcheck operand #" << en.index()
                              << " with numTransposeEliminated = "
                              << numTransposeEliminated << "\n");
      auto& operandPerm = operandPermMap[en.index()];
      if (isIdentityPermutation(operandPerm)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\t\tskip propagate for operand #" << en.index()
                   << " due to operandPerm is identity\n");
        continue;
      }
      auto operandDefiningOp = en.value().getDefiningOp();
      // transpose can be folded for const op.
      if (dyn_cast_or_null<mhlo::ConstantOp>(operandDefiningOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\t\tskip propagate for operand #" << en.index()
                   << " due to operand op is const op\n");
        continue;
      }
      bool needInsertTransposeOp = tryUpdate(en.value(), curOp, operandPerm);
      // Not support in-consistent permutation requirement for the same op
      // a.t.m.
      // TODO(kevin.zwy): support multiple different transpose requirements.
      if (!consistentPerm) {
        LLVM_DEBUG(llvm::dbgs() << "\t\tin-consistent permutation requirement "
                                   "detected, abort early\n");
        return success();
      }

      if (needInsertTransposeOp) {
        --numTransposeEliminated;
        LLVM_DEBUG(llvm::dbgs()
                   << "\t\tinsert transpose for operand #" << en.index()
                   << ", and --numTransposeEliminated = "
                   << numTransposeEliminated << "\n");
      }
    }
  }

  if (numTransposeEliminated > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "numTransposeEliminated = " << numTransposeEliminated
               << ", try to apply the move up\n\n");
    if (failed(ctx.rewriteIntermediateOps(permMap))) return failure();
    changed = true;
    return success();
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "numTransposeEliminated = " << numTransposeEliminated
               << ", skip to apply the move up\n\n");
  }
  return success();
}

LogicalResult tryMoveUpSingleTransposeGreedilyAndApplyIfBeneficial(
    TransposeSimpliferContext& ctx, bool& changed) {
  LLVM_DEBUG(llvm::dbgs()
             << "in tryMoveUpSingleTransposeGreedilyAndApplyIfBeneficial\n");
  for (auto transposeOp : ctx.getTransposeOps()) {
    LLVM_DEBUG(llvm::dbgs()
               << "\ttry: " << *transposeOp.getOperation() << "\n");
    if (failed(tryMoveUpTransposeGreedilyAndApplyIfBeneficialImpl(
            ctx, {transposeOp}, changed)))
      return failure();
    if (changed) break;
  }
  return success();
}

LogicalResult tryMoveUpTwoSiblingTransposesGreedilyAndApplyIfBeneficial(
    TransposeSimpliferContext& ctx, bool& changed) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "in tryMoveUpTwoSiblingTransposesGreedilyAndApplyIfBeneficial\n");
  auto graph = ctx.buildProducerGraph();

  for (int i = 0; i < ctx.getTransposeOps().size(); ++i) {
    for (int j = 0; j < i; ++j) {
      auto lhsTransposeOp = ctx.getTransposeOps()[i];
      auto rhsTransposeOp = ctx.getTransposeOps()[j];
      Operation* lhsOp = lhsTransposeOp.getOperation();
      Operation* rhsOp = rhsTransposeOp.getOperation();
      auto& lhsProducers = graph[lhsOp];
      auto& rhsProducers = graph[rhsOp];

      // skip if rhsOp is one of the producers of lhsOp.
      if (llvm::find(lhsProducers, rhsOp->getResult(0)) != lhsProducers.end())
        continue;

      // Check lhs & rhs have common producers.
      bool hasCommonProducer = llvm::any_of(rhsProducers, [&](Value v) {
        return llvm::find(lhsProducers, v) != lhsProducers.end();
      });
      if (!hasCommonProducer) continue;

      LLVM_DEBUG(llvm::dbgs() << "\ttry sibling transpose pair:\n");
      LLVM_DEBUG(llvm::dbgs() << "\t\tlhs: " << *lhsOp << "\n");
      LLVM_DEBUG(llvm::dbgs() << "\t\trhs: " << *rhsOp << "\n");

      if (failed(tryMoveUpTransposeGreedilyAndApplyIfBeneficialImpl(
              ctx, {lhsTransposeOp, rhsTransposeOp}, changed)))
        return failure();
      if (changed) break;
    }
  }
  return success();
}

// Calcucalte the benefit if we move up a given transpose op as far as possible
// and apply the rewrite when it's beneficial.
LogicalResult tryMoveUpTransposeGreedilyAndApplyIfBeneficialOnBlock(
    Block* block, bool& changed) {
  TransposeSimpliferContext ctx(block);

  if (failed(
          tryMoveUpSingleTransposeGreedilyAndApplyIfBeneficial(ctx, changed))) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "failed to tryMoveUpSingleTransposeGreedilyAndApplyIfBeneficial\n");
    return failure();
  }
  if (changed) return success();

  if (failed(tryMoveUpTwoSiblingTransposesGreedilyAndApplyIfBeneficial(
          ctx, changed))) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "failed to "
           "tryMoveUpTwoSiblingTransposesGreedilyAndApplyIfBeneficial\n");
    return failure();
  }
  if (changed) return success();

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

int64_t getNumTransposeOps(func::FuncOp func) {
  int64_t numTransposeOps = 0;
  func.walk([&](mhlo::TransposeOp) { ++numTransposeOps; });
  return numTransposeOps;
}

void dumpModule(func::FuncOp func) {
  llvm::dbgs() << "///------------ begin dump module:\n";
  func->getParentOfType<ModuleOp>().dump();
  llvm::dbgs() << "///------------ end dump module:\n";
}

LogicalResult TransposeSimplifierPass::runOnBlock(Block* block) {
  bool changed;
  int64_t prevNumTransposeOps = getNumTransposeOps(getOperation());
  LLVM_DEBUG(llvm::dbgs() << "run TransposeSimplifierPass on "
                          << *block->getParent()->getParentOp() << "\n");
  do {
    changed = false;

    if (failed(runCanonicalizer(getOperation()))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to do clean up");
      return failure();
    }
    LLVM_DEBUG(dumpModule(getOperation()));

    int64_t numTransposeOps = getNumTransposeOps(getOperation());
    LLVM_DEBUG(llvm::dbgs()
               << "num of transpose decreases from " << prevNumTransposeOps
               << " to " << numTransposeOps << "\n");
    prevNumTransposeOps = numTransposeOps;

    if (failed(tryMoveUpTransposeGreedilyAndApplyIfBeneficialOnBlock(
            block, changed))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to move up transpose greedily");
      return failure();
    }
    LLVM_DEBUG(
        llvm::dbgs()
        << "run after tryMoveUpTransposeGreedilyAndApplyIfBeneficialOnBlock\n");
    if (changed) continue;

    if (failed(reverseBinaryOpsIfBeneficial(block, changed))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed to reverseBinaryOpsIfBeneficial transpose ops");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "run after reverseBinaryOpsIfBeneficial\n");
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
