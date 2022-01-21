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

// This file implements the logic to propagate some known shape information.
// The basic flow is shown as below:
//   loop until converged:
//     stage #1: rewrite locally (pattern based)
//     - run applyPatternsAndFoldGreedily(...), where patterns from:
//         MhloOpShapeRefinerPattern #A/../#Z
//         TensorOrShapeOpsRefinerPatterns/Other patterns (e.g. const folding)
//     - example mhlo shape refiner pattern:
//       original:
//         %1 = "mhlo.XXXOp"(%0) : (tensor<?x10xf32>) -> tensor<?x?xf32>
//       to:
//         %1 = "mhlo.XXXOp"(%0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
//         // convert to original shape to remain valid IR and rely on the
//         second
//         // stage to propagate such information globally.
//         %2 = tensor.cast %1 : tensor<?x10xf32> to tensor<?x?xf32>
//
//     stage #2: propagate shape information globally, examples are:
//      convert from:
//        func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<10xf32>) ->
//        tensor<?xf32> {
//          %0 = tensor.cast %arg1 : tensor<10xf32> to tensor<?xf32>
//          %1 = "mhlo.add"(%arg0, %0) : (tensor<?xf32>, tensor<?xf32>) ->
//          tensor<?xf32> return %1 : tensor<?xf32>
//        }
//      to:
//        func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) ->
//        tensor<10xf32> {
//          %1 = "mhlo.add"(%arg0, %0) : (tensor<10xf32>, tensor<10xf32>) ->
//          tensor<10xf32> return %1 : tensor<10xf32>
//        }
//     stage #3: apply symbolic shape optimizations, examples are:
//      - broadcast optimizations:
//        - %output = broadcast(%input, %shape) -> %output = %input if %input
//        and %output have the same shape.

#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"    // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {
namespace {

// convert:
//   %shape = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
//   %dim_size = tensor.extract %shape[%c0] : tensor<2xindex>
// to:
//   %dim_size = tensor.dim %0, %c0 : tensor<2xindex>
struct ExtractFromExtentTensorCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto shape_of_op = op.tensor().getDefiningOp<shape::ShapeOfOp>();
    if (!shape_of_op) return failure();
    Value index = op.indices().front();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, shape_of_op.getArg(), index);
    return success();
  }
};

// Canonicalizes
// %c4_i32 = constant 4 : i32
// %shape = tensor.from_elements %0, %c4_i32 : tensor<2xi32>
// %1 = "mhlo.dynamic_reshape"(%tensor, %shape)  -> tensor<?x?xf32>
//
// into:
//
// %c4_i32 = constant 4 : i32
// %shape = tensor.from_elements %0, %c4_i32 : tensor<2xi32>
// %t = "mhlo.dynamic_reshape"(%tensor, %shape)  -> tensor<?x4xf32>
// %2 = tensor.cast(%t) tensor<?x?xf32> -> tensor<?x4xf32>
//
struct DynamicReshapeOpPartialShapeInference
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern<mhlo::DynamicReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto output_shape =
        op.output_shape().getDefiningOp<tensor::FromElementsOp>();
    if (!output_shape) {
      return failure();
    }
    auto result_type = op.getResult().getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> result_dims(result_type.getRank());
    bool has_uninfered_static_dim = false;
    for (auto element : llvm::enumerate(output_shape.elements())) {
      int64_t new_value = -1;
      if (result_type.isDynamicDim(element.index())) {
        if (arith::ConstantIntOp constant_op =
                element.value().getDefiningOp<arith::ConstantIntOp>()) {
          new_value = constant_op.getValue().cast<IntegerAttr>().getInt();
        } else if (arith::ConstantIndexOp constant_op =
                       element.value()
                           .getDefiningOp<arith::ConstantIndexOp>()) {
          new_value = constant_op.getValue().cast<IntegerAttr>().getInt();
        }
      }

      if (new_value != -1) {
        has_uninfered_static_dim = true;
        result_dims[element.index()] = new_value;
      } else {
        result_dims[element.index()] = result_type.getDimSize(element.index());
      }
    }
    if (!has_uninfered_static_dim) {
      return failure();
    }
    auto new_type = result_type.clone(result_dims);
    auto new_op = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, new_type, op.operand(), output_shape);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), new_op);
    return success();
  }
};

// Canonicalizes
// %cst_shape = constant dense<[a, b, .. c]> : tensor<nxi32>
// %0 = "mhlo.dynamic_reshape"(%tensor, %cst_shape)  -> tensor<?x?x?xf32>
//
// into:
//
// %cst_shape = constant dense<[a, b, .. c]> : tensor<nxi32>
// %t = "mhlo.dynamic_reshape"(%tensor, %cst_shape)  -> tensor<axbxcxf32>
// %1 = tensor.cast(%t) tensor<?x?x?xf32> -> tensor<axbxcxf32>
class DynamicReshapeOpShapeInference
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* shape_def_op = op.output_shape().getDefiningOp();
    if (!shape_def_op) return failure();
    DenseIntElementsAttr cst_attr;
    if (auto cst_shape = dyn_cast<arith::ConstantOp>(shape_def_op)) {
      cst_attr = cst_shape.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    } else if (auto mhlo_cst_shape = dyn_cast<mhlo::ConstOp>(shape_def_op)) {
      cst_attr =
          mhlo_cst_shape.value().dyn_cast_or_null<DenseIntElementsAttr>();
    }
    if (!cst_attr) return failure();
    auto elem_ty = cst_attr.getType().cast<ShapedType>().getElementType();
    SmallVector<int64_t, 4> dims;
    if (elem_ty.isInteger(64) || elem_ty.isIndex()) {
      std::copy(cst_attr.getValues<int64_t>().begin(),
                cst_attr.getValues<int64_t>().end(), std::back_inserter(dims));
    } else if (elem_ty.isInteger(32)) {
      std::copy(cst_attr.getValues<int32_t>().begin(),
                cst_attr.getValues<int32_t>().end(), std::back_inserter(dims));
    } else {
      return failure();
    }
    auto result_ty = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) return failure();
    RankedTensorType new_ty =
        RankedTensorType::get(dims, result_ty.getElementType());
    if (new_ty == result_ty) return failure();
    auto new_reshape = rewriter.create<mhlo::DynamicReshapeOp>(
        op.getLoc(), new_ty, op.operand(), op.output_shape());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), new_reshape);
    return success();
  }
};

// Match following patterns:
//  %0 = ...: tensor<?x?xf32>
//  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//  %new_shape = tensor.from_elements %d0, %d1, %c1 : tensor<3xindex>
//  %1 = "mhlo.dynamic_reshape"(%0, %new_shape) : (tensor<?x?xf32>,
//  tensor<3xindex>) -> tensor<?x?x1xf32> %2 =
//  "mhlo.dynamic_broadcast_in_dim"(%1, %...) {broadcast_dimensions = dense<[0,
//  1, 2]> : tensor<3xi64>} : (tensor<?x?x1xf32>, tensor<3xindex>) ->
//  tensor<?x?x?xf32>
// and convert to:
//  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %...) {broadcast_dimensions =
//  dense<[0, 1]> : tensor<3xi64>} : (tensor<?x?xf32>, tensor<3xindex>) ->
//  tensor<?x?x?xf32>
class DynamicBroadcastInDimOpSimplifier
    : public OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto reshapeOp = dyn_cast_or_null<mhlo::DynamicReshapeOp>(
        op->getOperand(0).getDefiningOp());
    if (!reshapeOp) return failure();

    auto bcastTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    auto reshapeTy =
        reshapeOp.getResult().getType().dyn_cast<RankedTensorType>();
    auto inputTy =
        reshapeOp->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!bcastTy || !reshapeTy || !inputTy) return failure();

    if (bcastTy.getRank() != reshapeTy.getRank() ||
        inputTy.getRank() >= reshapeTy.getRank())
      return failure();

    auto fromElementsOp = dyn_cast_or_null<tensor::FromElementsOp>(
        reshapeOp->getOperand(1).getDefiningOp());
    if (!fromElementsOp) return failure();

    SmallVector<int64_t> bcastDims;
    for (int d = 0; d < inputTy.getRank(); ++d) {
      Value dimValue = fromElementsOp->getOperand(d);
      auto indexCastOp =
          dyn_cast_or_null<arith::IndexCastOp>(dimValue.getDefiningOp());
      if (indexCastOp) dimValue = indexCastOp->getOperand(0);
      auto dimOp = dyn_cast_or_null<tensor::DimOp>(dimValue.getDefiningOp());
      if (!dimOp || dimOp.source() != reshapeOp->getOperand(0))
        return failure();
      auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(
          dimOp.index().getDefiningOp());
      if (!indexOp || indexOp.getValue().cast<IntegerAttr>().getInt() != d)
        return failure();
      bcastDims.push_back(d);
    }

    RankedTensorType ty = RankedTensorType::get(
        {static_cast<int64_t>(bcastDims.size())}, rewriter.getIntegerType(64));
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, op.getType(), reshapeOp->getOperand(0), op->getOperand(1),
        DenseIntElementsAttr::get(ty, bcastDims));

    return success();
  }
};

// Represents a symbolic dimension.
class SymbolDim {
 public:
  SymbolDim(int64_t dim_size = ShapedType::kDynamicSize) : dimSize_(dim_size) {}

  int64_t getDimSize() { return dimSize_; }

  bool isDynamic() { return getDimSize() == ShapedType::kDynamicSize; }

  LogicalResult Merge(SymbolDim* other);

 private:
  int64_t dimSize_;
};

// Merge two symbolDim if they are compatible.
LogicalResult SymbolDim::Merge(SymbolDim* other) {
  if (!isDynamic() && !other->isDynamic() &&
      getDimSize() != other->getDimSize())
    return failure();
  if (isDynamic()) dimSize_ = other->getDimSize();
  return success();
}

// Represents a symbolic ranked shape.
class SymbolShape {
 public:
  explicit SymbolShape(SmallVector<SymbolDim*, 4> dims = {})
      : dims_(std::move(dims)) {}

  int rank() { return dims_.size(); }

  void setSymbolDims(SmallVector<SymbolDim*, 4> dims) {
    dims_ = std::move(dims);
  }

  ArrayRef<SymbolDim*> getSymbolDims() { return dims_; }

  SmallVector<int64_t, 4> getDimValues() {
    SmallVector<int64_t, 4> dimValues;
    for (SymbolDim* dim : dims_) dimValues.push_back(dim->getDimSize());
    return dimValues;
  }

  SymbolDim* getSymbolDim(int dim) {
    assert(dim < dims_.size());
    return dims_[dim];
  }

  void setSymbolDim(int dim, SymbolDim* symbolDim) {
    assert(dim < dims_.size());
    dims_[dim] = symbolDim;
  }

  bool operator==(const SymbolShape& other) { return other.dims_ == dims_; }

 private:
  SmallVector<SymbolDim*, 4> dims_;
};

// A simple shape analysis for propagating known shape information in
// compilation time.
class ShapeAnalysis {
 public:
  explicit ShapeAnalysis(FuncOp func) : func_(func) {}

  LogicalResult run();

  FuncOp getFunc() { return func_; }

  Type getRefinedType(Value value);

  LogicalResult mapValueShapeEqual(Value lhs, Value rhs);
  LogicalResult mapValueDimEqual(Value lhs, int lhsDim, Value rhs, int rhsDim);

  bool isValueDimEqual(Value lhs, int lhsDim, Value rhs, int rhsDim);
  bool isValueShapeEqual(Value lhs, Value rhs);

  // Insert tie_shape ops to explicit tie dimension equality in the IR level.
  LogicalResult buildTieShapeOps();

 private:
  LogicalResult buildShapeMap();
  LogicalResult buildValueShape(Value value);
  LogicalResult buildRegionShapeMap(Region* region);
  LogicalResult buildBlockShapeMap(Block* block);
  LogicalResult buildOperationShapeMap(Operation* op);

  LogicalResult applyOpConstraint(Operation* op);
  LogicalResult applyTensorOpConstraint(Operation* op);
  LogicalResult applyMhloOpConstraint(Operation* op);

  SymbolDim* getRootDim(SymbolDim* symbolDim);
  SymbolDim* getDim(Value value, int dim);
  SymbolShape* getShape(Value value);

  LogicalResult mapDimEqual(SymbolDim* lhs, SymbolDim* rhs);
  LogicalResult mapShapeEqual(SymbolShape* lhs, SymbolShape* rhs);

  using SymbolShapeVisitor =
      std::function<LogicalResult(OpBuilder&, Location&, Value value)>;
  using Symbol2InstancesDominantType =
      DenseMap<SymbolDim*, DenseMap<Value, Value>>;
  LogicalResult visitSymbolShapes(SymbolShapeVisitor visitor);
  LogicalResult buildSymbolDimInstances(
      DenseMap<SymbolDim*, SmallVector<Value>>& symbolDim2Instances,
      DenseMap<Value, SmallVector<Value>>& shapedValue2DimValues);
  LogicalResult buildSymbolDimInstancesDominantMap(
      DenseMap<SymbolDim*, SmallVector<Value>>& instanceMap,
      DenseMap<SymbolDim*, DenseMap<Value, Value>>& dominantMap);
  void dumpSymbol2InstancesDominant(
      Symbol2InstancesDominantType symbol2InstancesDominant);

 private:
  bool initialized = false;
  FuncOp func_;
  SmallVector<std::unique_ptr<SymbolDim>, 4> dimVec_;
  DenseMap<SymbolDim*, int> dimMap_;
  DenseMap<Value, SymbolShape> shapeMap_;
};

Type ShapeAnalysis::getRefinedType(Value value) {
  SymbolShape* symbolShape = getShape(value);

  // not-shaped type
  if (!symbolShape) return value.getType();

  auto ty = value.getType().cast<RankedTensorType>();
  auto dimValues = symbolShape->getDimValues();
  return RankedTensorType::get(dimValues, ty.getElementType());
}

LogicalResult ShapeAnalysis::run() {
  // Make sure only run once.
  if (initialized) {
    func_.emitError() << "re-initialized shape analysis";
    return failure();
  }
  initialized = true;

  return buildShapeMap();
}

LogicalResult ShapeAnalysis::buildShapeMap() {
  return buildRegionShapeMap(&func_.getBody());
}

LogicalResult ShapeAnalysis::buildValueShape(Value value) {
  auto ty = value.getType().dyn_cast<ShapedType>();
  // Skip non-shaped value
  if (!ty) return success();

  // Not support dynamic rank a.t.m.
  if (!ty.hasRank()) return failure();

  SmallVector<SymbolDim*, 4> dims;
  for (int64_t dimSize : ty.getShape()) {
    dimVec_.emplace_back(new SymbolDim(dimSize));
    dims.push_back(dimVec_.back().get());
    dimMap_[dims.back()] = dimVec_.size() - 1;
  }

  shapeMap_[value] = SymbolShape(std::move(dims));
  return success();
}

LogicalResult ShapeAnalysis::buildRegionShapeMap(Region* region) {
  // Only SCF is supported a.t.m.
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitError(
        "only single block region is supported");
  }
  for (Block& block : *region) {
    if (failed(buildBlockShapeMap(&block))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysis::buildBlockShapeMap(Block* block) {
  // mapping block arguments
  for (Value value : block->getArguments()) {
    if (failed(buildValueShape(value))) {
      return block->getParentOp()->emitError(
          "failed to build shape for block arg");
    }
  }

  // mapping each op inside the block
  for (Operation& op : block->getOperations()) {
    if (failed(buildOperationShapeMap(&op))) return failure();
  }

  return success();
}

LogicalResult ShapeAnalysis::buildOperationShapeMap(Operation* op) {
  // build shapes for the results of op
  for (Value result : op->getResults()) {
    if (failed(buildValueShape(result))) {
      return op->emitError("failed to build shape for op's result");
    }
  }

  // apply op's shape constraint
  return applyOpConstraint(op);
}

SymbolShape* ShapeAnalysis::getShape(Value value) {
  auto it = shapeMap_.find(value);
  if (it == shapeMap_.end()) return nullptr;

  for (int i = 0; i < it->second.rank(); ++i) {
    it->second.setSymbolDim(i, getDim(value, i));
  }
  return &it->second;
}

SymbolDim* ShapeAnalysis::getRootDim(SymbolDim* symbolDim) {
  assert(symbolDim != nullptr);
  SymbolDim* parentSymbolDim = symbolDim;
  do {
    symbolDim = parentSymbolDim;
    int dimIdx = dimMap_[symbolDim];
    parentSymbolDim = dimVec_[dimIdx].get();
  } while (parentSymbolDim != symbolDim);

  return parentSymbolDim;
}

SymbolDim* ShapeAnalysis::getDim(Value value, int dim) {
  auto it = shapeMap_.find(value);
  if (it == shapeMap_.end()) return nullptr;

  SymbolDim* symbolDim = it->second.getSymbolDim(dim);
  assert(symbolDim != nullptr);

  symbolDim = getRootDim(symbolDim);
  it->second.setSymbolDim(dim, symbolDim);

  return symbolDim;
}

LogicalResult ShapeAnalysis::mapDimEqual(SymbolDim* lhs, SymbolDim* rhs) {
  int lhsIdx = dimMap_[getRootDim(lhs)];
  int rhsIdx = dimMap_[getRootDim(rhs)];

  lhs = dimVec_[lhsIdx].get();
  rhs = dimVec_[rhsIdx].get();

  // let the root with smaller idx to be the root of the merged group.
  if (lhsIdx <= rhsIdx) {
    dimMap_[rhs] = lhsIdx;
    return lhs->Merge(rhs);
  } else {
    dimMap_[lhs] = rhsIdx;
    return rhs->Merge(lhs);
  }
}

LogicalResult ShapeAnalysis::mapShapeEqual(SymbolShape* lhs, SymbolShape* rhs) {
  if (!lhs || !rhs || lhs->rank() != rhs->rank()) return failure();

  for (auto&& en : llvm::zip(lhs->getSymbolDims(), rhs->getSymbolDims())) {
    if (failed(mapDimEqual(std::get<0>(en), std::get<1>(en)))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysis::mapValueShapeEqual(Value lhs, Value rhs) {
  SymbolShape* lhsShape = getShape(lhs);
  SymbolShape* rhsShape = getShape(rhs);
  return mapShapeEqual(lhsShape, rhsShape);
}

LogicalResult ShapeAnalysis::mapValueDimEqual(Value lhs, int lhsIdx, Value rhs,
                                              int rhsIdx) {
  SymbolDim* lhsDim = getDim(lhs, lhsIdx);
  SymbolDim* rhsDim = getDim(rhs, rhsIdx);
  return mapDimEqual(lhsDim, rhsDim);
}

bool ShapeAnalysis::isValueDimEqual(Value lhs, int lhsDim, Value rhs,
                                    int rhsDim) {
  return getDim(lhs, lhsDim) == getDim(rhs, rhsDim);
}

bool ShapeAnalysis::isValueShapeEqual(Value lhs, Value rhs) {
  SymbolShape* lhsShape = getShape(lhs);
  SymbolShape* rhsShape = getShape(rhs);
  if (!lhsShape || !lhsShape) return false;
  return *lhsShape == *rhsShape;
}

LogicalResult ShapeAnalysis::applyOpConstraint(Operation* op) {
  if (op->getDialect() == op->getContext()->getLoadedDialect("tensor")) {
    return applyTensorOpConstraint(op);
  } else if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo") ||
             op->getDialect() ==
                 op->getContext()->getLoadedDialect("mhlo_disc")) {
    return applyMhloOpConstraint(op);
  }

  return success();
}

LogicalResult ShapeAnalysis::applyTensorOpConstraint(Operation* op) {
  if (isa<tensor::CastOp>(op)) {
    return mapValueShapeEqual(op->getResult(0), op->getOperand(0));
  }
  return success();
}

LogicalResult ShapeAnalysis::applyMhloOpConstraint(Operation* op) {
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() ||
      op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
    Value ref;
    if (op->getNumOperands() > 0) ref = op->getOperands().front();
    if (op->getNumResults() > 0) ref = op->getResults().front();
    if (!ref) return success();
    for (Value operand : op->getOperands()) mapValueShapeEqual(ref, operand);
    for (Value result : op->getResults()) mapValueShapeEqual(ref, result);
    return success();
  }

  if (op->hasTrait<mlir::OpTrait::SameTypeOperands>()) {
    if (op->getNumOperands() == 0) return success();
    Value ref = op->getOperands().front();
    for (Value operand : op->getOperands()) mapValueShapeEqual(ref, operand);
  }

  if (auto transpose = dyn_cast<mhlo::TransposeOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getResult(0);
    for (auto& en :
         llvm::enumerate(transpose.permutation().getValues<int64_t>())) {
      mapValueDimEqual(operand, en.value(), result, en.index());
    }
  } else if (auto concat = dyn_cast<mhlo::ConcatenateOp>(op)) {
    Value result = op->getResult(0);
    int64_t axis = concat.dimension();
    int64_t rank = result.getType().cast<RankedTensorType>().getRank();
    for (Value operand : op->getOperands()) {
      for (int64_t i = 0; i < rank; ++i) {
        if (i == axis) continue;
        mapValueDimEqual(operand, i, result, i);
      }
    }
  } else if (auto reduce = dyn_cast<mhlo::ReduceOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getResult(0);
    int64_t rank = operand.getType().cast<RankedTensorType>().getRank();
    int resultDimIdx = 0;
    for (int i = 0; i < rank; ++i) {
      auto reduceDims = reduce.dimensions().getValues<int64_t>();
      if (std::find(reduceDims.begin(), reduceDims.end(), i) !=
          reduceDims.end()) {
        continue;
      }
      mapValueDimEqual(operand, i, result, resultDimIdx++);
    }
  } else if (auto broadcast_in_dim =
                 dyn_cast<mhlo::DynamicBroadcastInDimOp>(op)) {
    auto operand = broadcast_in_dim->getOperand(0);
    Value result = op->getResult(0);
    auto ty = operand.getType().dyn_cast<ShapedType>();
    assert(ty);
    for (auto& dim : llvm::enumerate(
             broadcast_in_dim.broadcast_dimensions().getValues<int64_t>())) {
      // Deal with non-static & non-one dim.
      if (!ty.isDynamicDim(dim.index()) && ty.getDimSize(dim.index()) != 1) {
        mapValueDimEqual(operand, dim.index(), result, dim.value());
      }
    }
  } else if (auto dot_general = dyn_cast<mhlo::DotGeneralOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto dim_numbers = dot_general.dot_dimension_numbers();
    std::unordered_set<int64_t> lhs_contract_batch_dims;
    std::unordered_set<int64_t> rhs_contract_batch_dims;
    // Contracting dimensions.
    auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    assert(lhs_contracting_dims.size() == rhs_contracting_dims.size());
    for (int64_t i = 0; i < lhs_contracting_dims.size(); i++) {
      int64_t lhs_dim = lhs_contracting_dims[i];
      int64_t rhs_dim = rhs_contracting_dims[i];
      mapValueDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Batching dimensions.
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    assert(lhs_batching_dims.size() == rhs_batching_dims.size());
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      int64_t rhs_dim = rhs_batching_dims[i];
      mapValueDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Resulting dimensions. It follows that the resulting dimension number
    // starts with the batch dimension, then the 'lhs' non-contracting/non-batch
    // dimension, and finally the 'rhs' non-contracting/non-batch dimension.
    Value result = op->getResult(0);
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      mapValueDimEqual(lhs, lhs_dim, result, i);
    }
    SmallVector<std::pair<Value, int64_t>, 4> mn_values;
    for (int64_t i = 0; i < lhs.getType().cast<RankedTensorType>().getRank();
         i++) {
      if (lhs_contract_batch_dims.find(i) == lhs_contract_batch_dims.end()) {
        mn_values.emplace_back(lhs, i);
      }
    }
    for (int64_t i = 0; i < rhs.getType().cast<RankedTensorType>().getRank();
         i++) {
      if (rhs_contract_batch_dims.find(i) == rhs_contract_batch_dims.end()) {
        mn_values.emplace_back(rhs, i);
      }
    }
    for (int64_t i = 0; i < mn_values.size(); i++) {
      mapValueDimEqual(mn_values[i].first, mn_values[i].second, result,
                       i + lhs_batching_dims.size());
    }
  } else if (auto clamp = dyn_cast<mhlo::ClampOp>(op)) {
    int64_t min_rank = clamp.min().getType().cast<RankedTensorType>().getRank();
    int64_t max_rank = clamp.max().getType().cast<RankedTensorType>().getRank();
    if (min_rank != 0) {
      mapValueShapeEqual(clamp.operand(), clamp.min());
    }
    if (max_rank != 0) {
      mapValueShapeEqual(clamp.operand(), clamp.max());
    }
    mapValueShapeEqual(clamp.operand(), op->getResult(0));
  } else if (auto select = dyn_cast<mhlo::SelectOp>(op)) {
    int64_t pred_rank =
        select.pred().getType().cast<RankedTensorType>().getRank();
    int64_t true_rank =
        select.on_true().getType().cast<RankedTensorType>().getRank();
    int64_t false_rank =
        select.on_false().getType().cast<RankedTensorType>().getRank();
    if (pred_rank != 0) {
      mapValueShapeEqual(select.pred(), select.getResult());
    }
    if (true_rank != 0) {
      mapValueShapeEqual(select.on_true(), select.getResult());
    }
    if (false_rank != 0) {
      mapValueShapeEqual(select.on_false(), select.getResult());
    }
  }

  return success();
}

LogicalResult ShapeAnalysis::visitSymbolShapes(SymbolShapeVisitor visitor) {
  for (auto it = shapeMap_.begin(), stop = shapeMap_.end(); it != stop; ++it) {
    Value value = it->first;
    SymbolShape& symbolShape = it->second;
    OpBuilder b(func_);
    Location loc = func_.getLoc();
    Operation* definingOp = value.getDefiningOp();
    if (!definingOp) {
      Block* block = cast<BlockArgument>(value).getOwner();
      loc = block->getParentOp()->getLoc();
      b.setInsertionPoint(block, block->begin());
    } else {
      loc = definingOp->getLoc();
      b.setInsertionPointAfter(definingOp);
    }
    if (failed(visitor(b, loc, value))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysis::buildSymbolDimInstances(
    DenseMap<SymbolDim*, SmallVector<Value>>& symbolDim2Instances,
    DenseMap<Value, SmallVector<Value>>& shapedValue2DimValues) {
  // Instantiate each symbol dim and group all instances for each symbol dim.
  SymbolShapeVisitor visitor = [&](OpBuilder& b, Location& loc, Value value) {
    SymbolShape* symbolShape = getShape(value);
    auto& dimValues = shapedValue2DimValues[value];
    for (int i = 0, rank = symbolShape->rank(); i < rank; ++i) {
      Value dimSize = b.create<tensor::DimOp>(loc, value, i);
      SymbolDim* symbolDim = symbolShape->getSymbolDim(i);
      assert(symbolDim);
      auto& instances = symbolDim2Instances[symbolDim];
      instances.push_back(dimSize);
      dimValues.push_back(dimSize);
    }
    return success();
  };
  return visitSymbolShapes(visitor);
}

LogicalResult ShapeAnalysis::buildSymbolDimInstancesDominantMap(
    DenseMap<SymbolDim*, SmallVector<Value>>& instanceMap,
    DenseMap<SymbolDim*, DenseMap<Value, Value>>& dominantMap) {
  DominanceInfo dominanceInfo(func_);
  for (auto& it : instanceMap) {
    auto& symbolDim = it.first;
    auto& instances = it.second;
    auto& dominants = dominantMap[symbolDim];

    // in normal cases, there should be only one root, aka, dominant value
    // of all the values of instances
    SmallVector<Value> roots;
    for (Value v : instances) {
      bool is_root = true;
      for (Value other : instances) {
        if (v == other) continue;
        if (dominanceInfo.dominates(other, v.getDefiningOp())) {
          is_root = false;
          continue;
        }
      }
      if (is_root) {
        roots.push_back(v);
      }
    }
    // we should let as much values as possible to be dominated by a same root
    for (Value root : roots) {
      for (Value v : instances) {
        if (dominants.find(v) == dominants.end() &&
            dominanceInfo.dominates(root, v.getDefiningOp())) {
          dominants[v] = root;
        }
      }
    }
    assert(dominants.size() == instances.size());
  }
  return success();
}

void ShapeAnalysis::dumpSymbol2InstancesDominant(
    Symbol2InstancesDominantType symbol2InstancesDominant) {
  llvm::dbgs() << "symbol2InstancesDominant: "
               << symbol2InstancesDominant.size() << "\n";
  for (auto item : symbol2InstancesDominant) {
    llvm::dbgs() << " -- item: \n";
    for (auto val_pair : item.second) {
      llvm::dbgs() << " ---- " << val_pair.first << " map to "
                   << val_pair.second << "\n";
    }
  }
}

// After we do all shape-related simplifications in the tensor level, we build
// explicit connection between all symbolic dims using 'disc_shape.tie_shape'
// right before we go to the buffer world.
//
// The basic idea is shown as below:
// Example IR before processing:
//   func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = mhlo.add(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) ->
//     tensor<?xf32> return %0 : tensor<?xf32>
//   }
// After adding disc.tie_shape ops:
//   func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = tensor.dim %arg0, %c0
//     %new_arg0 = disc_shape.tie_shape(%arg0, %arg0_d0)
//     %new_arg1 = disc_shape.tie_shape(%arg1, %arg0_d0)
//     %0 = mhlo.add(%new_arg0, %new_arg1) : (tensor<?xf32>, tensor<?xf32>) ->
//     tensor<?xf32>
//
//     %new_0 = disc_shape.tie_shape(%0, %arg0_d0)
//     return %new_0 : tensor<?xf32>
//   }
// The disc.tie_shape op is translated to a memref.reinterpret_cast op when
// converting to the buffer world. Example IR after bufferizing:
//   func @main(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> memref<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = memref.dim %arg0, %c0
//     %new_arg0 = memref.reinterpret_cast %arg0 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %new_arg1 = memref.reinterpret_cast %arg1 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %0 = memref.alloc(%arg0_d0) : memref<?xf32>
//     "lmhlo.add"(%new_arg0, %new_arg1, %0) : (memref<?xf32>, ...
//
//     %new_0 = memref.reinterpret_cast %0 to
//     offset: [0], sizes: [%arg0_d0], strides: [1] : memref<?xf32> to
//     memref<?xf32>
//
//     return %new_0 : tensor<?xf32>
//   }
// After doing AllocOp + reinterpret_cast op canonicalization:
//   func @main(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> memref<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = memref.dim %arg0, %c0
//     %new_arg0 = memref.reinterpret_cast %arg0 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %new_arg1 =
//     memref.reinterpret_cast %arg1 to offset: [0], sizes: [%arg0_d0], strides:
//     [1] : memref<?xf32> to memref<?xf32>
//
//     %0 = memref.alloc(%arg0_d0) : memref<?xf32>
//     "lmhlo.add"(%new_arg0, %new_arg1, %0) : (memref<?xf32>, ...
//     return %0 : memref<?xf32>
//   }
// After the above processing, symbolic dims are resolved into the same SSA
// value if they are the same (e.g. the operands for allocOp or reinterpret_cast
// itself if it's external buffer) and following passes can simply reply on
// normal CSE & canonicalization pass to simplify index computation.
LogicalResult ShapeAnalysis::buildTieShapeOps() {
  // create a dim op for each dimension of a shaped Value and group all such dim
  // ops by SymbolDim. Examples:
  //   %1 = mhlo.abs(%0) : (tensor<?xf32>) -> tensor<?xf32>
  // After processing:
  //   %0_d0 = tensor.dim %0, c0
  //   %1 = mhlo.abs(%0) : (tensor<?xf32>) -> tensor<?xf32>
  //   %1_d0 = tensor.dim %1, c0
  // symbolDim2Instances:
  //   {symbol0 : {%0_d0, %1_d0}} // %0_d0 and %1_d0 have the same symbolDim
  // shapedValue2DimValues:
  //   {%0 : {%0_d0}, %1 : {%1_d0}}
  DenseMap<SymbolDim*, SmallVector<Value>> symbolDim2Instances;
  DenseMap<Value, SmallVector<Value>> shapedValue2DimValues;
  if (failed(
          buildSymbolDimInstances(symbolDim2Instances, shapedValue2DimValues)))
    return func_->emitError("failed to buildSymbolDimInstances");

  // map SymbolDim to its instance dominance map.
  // instance dominance map: instance -> dominant instance
  Symbol2InstancesDominantType symbol2InstancesDominant;
  if (failed(buildSymbolDimInstancesDominantMap(symbolDim2Instances,
                                                symbol2InstancesDominant)))
    return func_->emitError("failed to buildSymbolDimInstancesDominantMap");

  LLVM_DEBUG(dumpSymbol2InstancesDominant(symbol2InstancesDominant));

  // create a tie_shape op for each shaped value.
  SymbolShapeVisitor visitor = [&](OpBuilder& b, Location& loc, Value value) {
    // Skip static shaped values
    if (value.getType().cast<RankedTensorType>().hasStaticShape()) {
      return success();
    }

    SmallVector<Value> dominantDimValues;
    SymbolShape* symbolShape = getShape(value);
    auto& dimValues = shapedValue2DimValues[value];
    DenseSet<Operation*> dimOps;
    for (int i = 0, rank = symbolShape->rank(); i < rank; ++i) {
      SymbolDim* symbolDim = symbolShape->getSymbolDim(i);
      assert(symbolDim);
      auto& dominantInfo = symbol2InstancesDominant[symbolDim];
      dominantDimValues.push_back(dominantInfo[dimValues[i]]);
      // if 'value' is not an BlockArgument, we can guarantee
      // 'dominantDimValues' dominate 'value'
      if (dimValues[i] == dominantDimValues.back() ||
          value.isa<BlockArgument>()) {
        b.setInsertionPointAfter(dimValues[i].getDefiningOp());
      }
      dimOps.insert(dimValues[i].getDefiningOp());
    }
    Value newValue = b.create<disc_shape::TieShapeOp>(loc, value.getType(),
                                                      value, dominantDimValues);
    auto users = llvm::to_vector<4>(value.getUsers());
    for (Operation* user : users) {
      // skip those dim ops used to fetch the dim size values of original shaped
      // value.
      if (dimOps.find(user) != dimOps.end()) continue;
      if (user == newValue.getDefiningOp()) continue;
      user->replaceUsesOfWith(value, newValue);
    }
    return success();
  };

  return visitSymbolShapes(visitor);
}

struct ShapeSimplifierPass
    : public DiscShapeSimplifierPassBase<ShapeSimplifierPass> {
  ShapeSimplifierPass(const std::string& entry_func_name, bool insert_tie_shape)
      : DiscShapeSimplifierPassBase<
            ShapeSimplifierPass>::DiscShapeSimplifierPassBase() {
    this->entry_func_name_ = entry_func_name;
    this->insert_tie_shape_ = insert_tie_shape;
  }

  // Adds canonicalization patterns to the list of patterns.
  void AddCanonicalizationPatterns(MLIRContext* context,
                                   OwningRewritePatternList* patterns) {
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(*patterns, context);
  }

  void populateShapeRefinerPatterns(OwningRewritePatternList&);

  void runOnOperation() override;

  LogicalResult applyShapeAnalysis(ShapeAnalysis&, bool&);

  LogicalResult applySymbolicShapeOptimization(ShapeAnalysis&, bool&);
};

void ShapeSimplifierPass::populateShapeRefinerPatterns(
    OwningRewritePatternList& patterns) {
  // clang-format off
  patterns.insert<
      ExtractFromExtentTensorCanonicalizationPattern,
      DynamicBroadcastInDimOpSimplifier,
      // TODO: upstream these general purpose patterns to hlo_ops.cc
      DynamicReshapeOpPartialShapeInference,
      DynamicReshapeOpShapeInference
  >(patterns.getContext());
  // clang-format on

  // Adds canonicalization patterns to the list of patterns.
  AddCanonicalizationPatterns(patterns.getContext(), &patterns);
}

void ShapeSimplifierPass::runOnOperation() {
  ModuleOp m = getOperation();
  FuncOp main = m.lookupSymbol<FuncOp>(entry_func_name_);
  if (!main) {
    m.emitError("entry func: " + entry_func_name_ + " not found");
    signalPassFailure();
    return;
  }

  bool changed = true;
  while (changed) {
    // suppose not change the IR by default.
    changed = false;

    // Stage #1: refine shape information locally
    // - Initialize local shape refiner patterns.
    OwningRewritePatternList patterns(main.getContext());
    populateShapeRefinerPatterns(patterns);
    // - apply these patterns
    // ignore the not-converged error since we are in a loop.
    (void)applyPatternsAndFoldGreedily(main, std::move(patterns));

    // Stage #2: propagate shape information globally
    ShapeAnalysis analysis(main);
    if (failed(analysis.run())) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    if (failed(applyShapeAnalysis(analysis, changed))) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    // Stage #3: apply symbolic shape optimization. e.g. %out = bcast(%in) ->
    // %out = %in
    if (failed(applySymbolicShapeOptimization(analysis, changed))) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    // In the last interation, we explicitly insert tie_shape op to connect
    // symbol equal dimensions in the IR level.
    if (!changed && insert_tie_shape_ && failed(analysis.buildTieShapeOps())) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }
  }
}

LogicalResult ShapeSimplifierPass::applyShapeAnalysis(ShapeAnalysis& analysis,
                                                      bool& changed) {
  FuncOp func = analysis.getFunc();

  auto updateIfNotSame = [&](Value value) {
    Type refinedTy = analysis.getRefinedType(value);
    if (refinedTy != value.getType()) {
      changed = true;
      value.setType(refinedTy);
    }
  };

  // apply refined type for each value
  func.walk([&](Operation* op) {
    for (Value operand : op->getOperands()) updateIfNotSame(operand);

    for (Value result : op->getResults()) updateIfNotSame(result);
  });

  // apply refined function type
  // 1, collect input types
  SmallVector<Type, 4> refinedInputTypes;
  for (Value arg : func.getArguments())
    refinedInputTypes.push_back(analysis.getRefinedType(arg));

  // 2, collect output types
  SmallVector<Type, 4> refinedOutputTypes;
  assert(func.getBody().getBlocks().size() == 1);
  Operation& op = func.getBody().front().getOperations().back();
  for (Value operand : op.getOperands())
    refinedOutputTypes.push_back(analysis.getRefinedType(operand));

  // 3, refine function type to new type
  auto newFuncTy = FunctionType::get(func.getContext(), refinedInputTypes,
                                     refinedOutputTypes);
  if (func.getType() != newFuncTy) {
    func.setType(newFuncTy);
    changed = true;
  }

  return success();
}

LogicalResult ShapeSimplifierPass::applySymbolicShapeOptimization(
    ShapeAnalysis& analysis, bool& changed) {
  FuncOp func = analysis.getFunc();

  // 1, %out = bcast(%in, ...) -> identity if shape_of(%in) == shape_of(%out)
  SmallVector<Operation*, 4> bcastOps;
  func.walk([&](mhlo::DynamicBroadcastInDimOp op) {
    if (analysis.isValueShapeEqual(op.getResult(), op->getOperand(0)))
      bcastOps.push_back(op);
  });
  for (Operation* op : bcastOps) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    changed = true;
  }

  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeSimplifierPass(
    const std::string& entry_func_name, bool insert_tie_shape) {
  return std::make_unique<ShapeSimplifierPass>(entry_func_name,
                                               insert_tie_shape);
}

}  // namespace disc_ral
}  // namespace mlir
