/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

// This file implements the logic to do some shape optimizations on tensor
// level.
#include <chrono>
#include <numeric>
#include <stack>
#include <unordered_set>
#include <utility>

#include "absl/strings/str_split.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;
namespace {
std::string kDynamicDimsAttr = "input_dynamic_dims";
struct ShapeContext {
  ShapeContext() = default;
  ShapeContext(Value value, SmallVector<int64_t> shape)
      : value(value), shape(shape){};
  Value value;
  SmallVector<int64_t> shape;
};
struct DiscShapePropagatePass
    : public DiscShapePropagatePassBase<DiscShapePropagatePass> {
  DiscShapePropagatePass()
      : DiscShapePropagatePassBase<
            DiscShapePropagatePass>::DiscShapePropagatePassBase() {}
  void getDependentDialects(DialectRegistry& registry) const override {
    DiscShapePropagatePassBase<DiscShapePropagatePass>::getDependentDialects(
        registry);
    registry.insert<shape::ShapeDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<mhlo::MhloDialect>();
  }
  void visitOperator(ModuleOp& m, OpBuilder& rewriter, Operation* op,
                     std::stack<ShapeContext>& ctxStack);
  void runOnOperation() override;
};
bool isBinaryOp(Operation* op) {
  return isa<mhlo::AddOp, mhlo::CompareOp, mhlo::SelectOp, mhlo::PowOp,
             mhlo::SubtractOp, mhlo::MulOp, mhlo::DivOp, mhlo::MaxOp>(op);
}

bool isUnaryOp(Operation* op) {
  return isa<mhlo::ConvertOp, mhlo::ScatterOp, mhlo::SqrtOp, mhlo::LogisticOp,
             mhlo::RsqrtOp, mhlo::NegOp, mhlo::ExpOp, mhlo::LogOp>(op);
}
bool isConcreteShape(ShapeContext& ctx) {
  for (auto dim : ctx.shape) {
    if (dim == ShapedType::kDynamic) return false;
  }
  return true;
}

std::optional<Value> getConstTensor(OpBuilder& b, Operation* op,
                                    ArrayRef<int> vec,
                                    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }
  auto const_type = RankedTensorType::get(shape, b.getI64Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);
  auto const_op =
      b.create<mhlo::ConstantOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

std::optional<ShapeContext> HandleBinaryOp(OpBuilder& b, Operation* op,
                                           ShapeContext& inputCtx) {
  auto bcastOp = dyn_cast_or_null<mhlo::BroadcastInDimOp>(
      op->getOperand(1).getDefiningOp());
  if (!bcastOp) {
    return ShapeContext(op->getResult(0), inputCtx.shape);
  }
  if (bcastOp) {
    auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(
        bcastOp->getOperand(0).getDefiningOp());
    if (!constOp) {
      return ShapeContext(op->getResult(0), inputCtx.shape);
    }
    auto elemTy =
        op->getOperand(0).getType().cast<RankedTensorType>().getElementType();
    b.setInsertionPoint(op);
    auto dense_attr = constOp.getValue().dyn_cast<mlir::DenseElementsAttr>();
    int64_t value = dense_attr.getValues<int64_t>()[0];
    auto scalar_const_op = getConstTensor(b, op, {value}, {});
    Value inputShape =
        b.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0));
    auto rank = inputCtx.shape.size();

    auto dynBcastOp = b.create<mhlo::DynamicBroadcastInDimOp>(
        op->getLoc(), RankedTensorType::get(inputCtx.shape, elemTy),
        scalar_const_op.value(), inputShape, b.getI64TensorAttr({}));
    bcastOp.getResult().replaceAllUsesWith(dynBcastOp.getResult());
  }
  return ShapeContext(op->getResult(0), inputCtx.shape);
}

template <typename OpTy>
std::optional<ShapeContext> propagateHelper(OpBuilder& b, Operation* op,
                                            ShapeContext& inputCtx) {
  return std::nullopt;
}

template <>
std::optional<ShapeContext> propagateHelper<tensor::DimOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto dim_op = dyn_cast_or_null<tensor::DimOp>(op);
  if (!dim_op) return std::nullopt;

  SmallVector<int64_t> new_shape(
      op->getResult(0).getType().cast<RankedTensorType>().getShape());
  return ShapeContext(op->getResult(0), new_shape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::DotOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto dot_op = dyn_cast<mhlo::DotOp>(op);
  if (!dot_op) return std::nullopt;
  auto lhs = dot_op.getOperand(0);
  auto rhs = dot_op.getOperand(1);
  if (inputCtx.value == lhs) {
    return ShapeContext(op->getResult(0),
                        {inputCtx.shape[0],
                         rhs.getType().cast<RankedTensorType>().getShape()[1]});
  } else {
    return ShapeContext(op->getResult(0),
                        {lhs.getType().cast<RankedTensorType>().getShape()[0],
                         inputCtx.shape[1]});
  }
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::ReshapeOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto reshape_op = dyn_cast<mhlo::ReshapeOp>(op);
  if (!reshape_op) return std::nullopt;
  Type intType = b.getIntegerType(32);
  int rank =
      reshape_op.getOperand().getType().cast<RankedTensorType>().getRank();
  auto resultRankType =
      reshape_op.getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultRankType.getRank();
  auto resultShape = resultRankType.getShape();
  SmallVector<int64_t> newShape(resultRank, ShapedType::kDynamic);
  int64_t numel =
      std::accumulate(inputCtx.shape.begin(), inputCtx.shape.end(), int64_t(1),
                      [](int64_t acc, int64_t num) {
                        return num == ShapedType::kDynamic ? acc : acc * num;
                      });

  bool inferenced = true;
  while (inferenced) {
    inferenced = false;
    // set concret shape if possible
    for (size_t i = 0; i < resultRank; ++i) {
      for (size_t j = 0; j < rank; ++j) {
        if (newShape[i] == ShapedType::kDynamic &&
            resultShape[i] == inputCtx.shape[j]) {
          newShape[i] = inputCtx.shape[j];
          numel /= inputCtx.shape[j];
          inferenced = true;
        }
      }
    }
    for (size_t d = 0; d < resultRank; ++d) {
      if (newShape[d] == ShapedType::kDynamic) {
        if (numel % resultShape[d] == 0) {
          numel /= resultShape[d];
          newShape[d] = resultShape[d];
          inferenced = true;
        }
      }
    }
  }
  // more then one dynamic dims is invalid, let's try to use the concret shape
  // to fill the dynamic dims
  int dynDims =
      std::count(newShape.begin(), newShape.end(), ShapedType::kDynamic);
  for (size_t i = 0; i < resultRank; ++i) {
    if (newShape[i] == ShapedType::kDynamic && dynDims > 1) {
      newShape[i] = resultShape[i];
      dynDims--;
      break;
    }
  }
  SmallVector<Value, 4> newShapeValues;
  for (int64_t dim : newShape) {
    if (dim == ShapedType::kDynamic) {
      // caculate the dimension
      newShapeValues.push_back(
          b.create<arith::ConstantIndexOp>(op->getLoc(), -1));
    } else {
      newShapeValues.push_back(
          b.create<arith::ConstantIndexOp>(op->getLoc(), dim));
    }
  }
  Value shapeValue =
      b.create<tensor::FromElementsOp>(op->getLoc(), newShapeValues);

  auto shape = b.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0));
  auto numElems = b.create<shape::NumElementsOp>(op->getLoc(), shape);

  auto computeReshapeShape = b.create<mhlo::ComputeReshapeShapeOp>(
      op->getLoc(), shapeValue.getType(), numElems.getResult(), shapeValue);
  auto dynReshapeOpResultType =
      RankedTensorType::get(newShape, resultRankType.getElementType());
  auto dynReshapeOp = b.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), dynReshapeOpResultType, reshape_op.getOperand(),
      computeReshapeShape);
  op->getResult(0).replaceAllUsesWith(dynReshapeOp.getResult());
  op->erase();
  return ShapeContext(dynReshapeOp->getResult(0), newShape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::SliceOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto slice_op = dyn_cast<mhlo::SliceOp>(op);
  if (!slice_op) return std::nullopt;
  b.setInsertionPoint(op);
  auto loc = slice_op.getLoc();
  auto rankType = slice_op.getOperand().getType().cast<RankedTensorType>();

  auto inputShape = rankType.getShape();
  auto rank = rankType.getRank();
  SmallVector<Value, 4> startIndices(rank);
  SmallVector<Value, 4> limitIndices(rank);
  SmallVector<Value, 4> strides(rank);
  SmallVector<int64_t> newShape(rank);
  for (size_t i = 0; i < rankType.getRank(); ++i) {
    auto startIndicesCst = slice_op.getStartIndices().getValues<int64_t>()[i];
    auto limitIndicesCst = slice_op.getLimitIndices().getValues<int64_t>()[i];
    auto stridesCst = slice_op.getStrides().getValues<int64_t>()[i];
    startIndices[i] =
        b.create<arith::ConstantIndexOp>(slice_op.getLoc(), startIndicesCst);
    // using dynamic dim if limitIndices is the same as input shape
    if (limitIndicesCst == inputShape[i] &&
        inputCtx.shape[i] == ShapedType::kDynamic) {
      limitIndices[i] = b.create<tensor::DimOp>(loc, slice_op.getOperand(), i);
      newShape[i] = inputCtx.shape[i];
    } else {
      limitIndices[i] =
          b.create<arith::ConstantIndexOp>(slice_op.getLoc(), limitIndicesCst);
      newShape[i] = (limitIndicesCst - startIndicesCst - 1) / stridesCst + 1;
    }
    strides[i] =
        b.create<arith::ConstantIndexOp>(slice_op.getLoc(), stridesCst);
  }
  Value baseIndicesValue = b.create<tensor::FromElementsOp>(loc, startIndices);
  Value stridesValue = b.create<tensor::FromElementsOp>(loc, strides);
  Value limitIndicesValue = b.create<tensor::FromElementsOp>(loc, limitIndices);
  auto sliceOpResultType =
      RankedTensorType::get(newShape, rankType.getElementType());
  auto dyncSliceOp = b.create<mhlo::RealDynamicSliceOp>(
      loc, sliceOpResultType, slice_op.getOperand(), baseIndicesValue,
      limitIndicesValue, stridesValue);
  op->getResult(0).replaceAllUsesWith(dyncSliceOp.getResult());
  op->erase();
  return ShapeContext(dyncSliceOp->getResult(0), newShape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::ConcatenateOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto concat_op = dyn_cast_or_null<mhlo::ConcatenateOp>(op);
  if (!concat_op) return std::nullopt;

  auto operands = op->getOperands();
  SmallVector<int64_t> new_shape(
      op->getResult(0).getType().cast<RankedTensorType>().getRank(),
      ShapedType::kDynamic);
  new_shape[concat_op.getDimension()] =
      op->getResult(0)
          .getType()
          .cast<RankedTensorType>()
          .getShape()[concat_op.getDimension()];

  for (auto operand : operands) {
    auto shape = operand.getType().cast<RankedTensorType>().getShape();
    if (inputCtx.value == operand) {
      shape = inputCtx.shape;
    }

    for (int dim_idx = 0; dim_idx < new_shape.size(); dim_idx++) {
      if (dim_idx == concat_op.getDimension() &&
          shape[dim_idx] == ShapedType::kDynamic) {
        new_shape[dim_idx] = ShapedType::kDynamic;
      } else if (dim_idx != concat_op.getDimension() &&
                 shape[dim_idx] != ShapedType::kDynamic) {
        new_shape[dim_idx] = shape[dim_idx];
      }
    }
  }

  return ShapeContext(op->getResult(0), new_shape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::TransposeOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto transpose_op = dyn_cast_or_null<mhlo::TransposeOp>(op);
  if (!transpose_op) return std::nullopt;

  SmallVector<int64_t> new_shape;

  for (auto it = transpose_op.getPermutation().begin();
       it != transpose_op.getPermutation().end(); it++) {
    int64_t src_dim = (*it).getSExtValue();
    new_shape.push_back(inputCtx.shape[src_dim]);
  }

  return ShapeContext(op->getResult(0), new_shape);
}
template <>
std::optional<ShapeContext> propagateHelper<mhlo::DotGeneralOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto dot_general_op = dyn_cast_or_null<mhlo::DotGeneralOp>(op);
  if (!dot_general_op) return std::nullopt;
  auto lhs = dot_general_op.getOperand(0);
  auto rhs = dot_general_op.getOperand(1);
  if (inputCtx.value == lhs) {
    return ShapeContext(op->getResult(0),
                        {rhs.getType().cast<RankedTensorType>().getShape()[0],
                         inputCtx.shape[1],
                         rhs.getType().cast<RankedTensorType>().getShape()[2]});
  } else {
    return ShapeContext(op->getResult(0),
                        {lhs.getType().cast<RankedTensorType>().getShape()[0],
                         lhs.getType().cast<RankedTensorType>().getShape()[1],
                         inputCtx.shape[2]});
  }
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::ReduceOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto reduce_op = dyn_cast_or_null<mhlo::ReduceOp>(op);
  if (!reduce_op) return std::nullopt;

  SmallVector<int64_t> new_shape;

  for (int dim = 0; dim < inputCtx.shape.size(); dim++) {
    bool add_dim = true;
    for (auto it = reduce_op.getDimensions().begin();
         it != reduce_op.getDimensions().end(); it++) {
      int64_t src_dim = (*it).getSExtValue();
      add_dim = add_dim && !(dim == src_dim);
    }
    if (add_dim) {
      new_shape.push_back(inputCtx.shape[dim]);
    }
  }

  return ShapeContext(op->getResult(0), new_shape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::DynamicGatherOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto dynamic_gather_op = dyn_cast_or_null<mhlo::DynamicGatherOp>(op);
  if (!dynamic_gather_op) return std::nullopt;

  SmallVector<int64_t> new_shape(dynamic_gather_op.getResult()
                                     .getType()
                                     .cast<RankedTensorType>()
                                     .getShape());

  auto attr = dynamic_gather_op.getDimensionNumbers();
  auto slice_sizes =
      op->getOperand(2).getType().cast<RankedTensorType>().getShape();

  auto offset_dims = attr.getOffsetDims();
  auto index_vector_dim = attr.getIndexVectorDim();
  auto collapsed_slice_dims = attr.getCollapsedSliceDims();

  if (inputCtx.value == op->getOperand(1)) {
    // start_indices
    int shape_dim_idx = 0;
    for (int dim_idx = 0; dim_idx < inputCtx.shape.size(); dim_idx++) {
      if (dim_idx != index_vector_dim) {
        new_shape[shape_dim_idx++] = inputCtx.shape[dim_idx];
      }
    }
  } else if (inputCtx.value == op->getOperand(2)) {
    int shape_dim_idx =
        op->getOperand(0).getType().cast<RankedTensorType>().getRank() - 1;
    for (int dim_idx = 0; dim_idx < inputCtx.shape.size(); dim_idx++) {
      bool include_this_dim = true;
      for (auto collapsed_slice_dim : collapsed_slice_dims) {
        if (dim_idx == collapsed_slice_dim) {
          include_this_dim = false;
        }
      }
      if (include_this_dim) {
        // need to decide whether it is a constant value or value from operand
        new_shape[shape_dim_idx++] = inputCtx.shape[dim_idx];
      }
    }
  }

  return ShapeContext(op->getResult(0), new_shape);
}
template <>
std::optional<ShapeContext> propagateHelper<mhlo::DynamicReshapeOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto resultShape =
      op->getResult(0).getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> newShape(resultShape.begin(), resultShape.end());
  return ShapeContext(op->getResult(0), newShape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::RealDynamicSliceOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto resultShape =
      op->getResult(0).getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> newShape(resultShape.begin(), resultShape.end());
  return ShapeContext(op->getResult(0), newShape);
}
template <>
std::optional<ShapeContext> propagateHelper<mhlo::DynamicBroadcastInDimOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto resultShape =
      op->getResult(0).getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> newShape(resultShape.begin(), resultShape.end());
  return ShapeContext(op->getResult(0), newShape);
}

template <>
std::optional<ShapeContext> propagateHelper<mhlo::GatherOp>(
    OpBuilder& b, Operation* op, ShapeContext& inputCtx) {
  auto gather_op = dyn_cast_or_null<mhlo::GatherOp>(op);
  if (!gather_op) return std::nullopt;

  // batch_dims = [d for d in axes(result) and d not in offset_dims].
  auto attr = gather_op.getDimensionNumbers();
  auto offset_dims = attr.getOffsetDims();
  auto index_vector_dim = attr.getIndexVectorDim();
  auto slice_sizes = gather_op.getSliceSizes();
  auto collapsed_slice_dims = attr.getCollapsedSliceDims();
  auto src_shape =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  SmallVector<Value> slice_sizes_vec;
  SmallVector<int64_t> new_shape;
  auto start_indices_shape =
      op->getOperand(1).getType().cast<RankedTensorType>().getShape();

  b.setInsertionPoint(op);
  // process offset_dim_sizes, offset dims
  for (int dim_idx = 0; dim_idx < start_indices_shape.size(); dim_idx++) {
    if (dim_idx != index_vector_dim) {
      new_shape.push_back(start_indices_shape[dim_idx]);
    }
  }

  int dim_idx = 0;
  for (auto dim_size : slice_sizes) {
    bool include_this_dim = true;
    for (auto collapsed_slice_dim : collapsed_slice_dims) {
      if (dim_idx == collapsed_slice_dim) {
        include_this_dim = false;
      }
    }
    // need to decide whether it is a constant value or value from operand
    if (src_shape[dim_idx] == dim_size.getSExtValue()) {
      auto dim_value = b.create<tensor::DimOp>(op->getLoc(), op->getOperand(0),
                                               b.create<arith::ConstantIndexOp>(
                                                    op->getLoc(), dim_idx)
                                                   .getResult())
                           .getResult();
      slice_sizes_vec.push_back(
          b.create<arith::IndexCastOp>(op->getLoc(), b.getI64Type(), dim_value)
              .getResult());
    } else {
      slice_sizes_vec.push_back(b.create<arith::ConstantIntOp>(
          op->getLoc(), dim_size.getSExtValue(), b.getI64Type()));
    }

    if (include_this_dim && src_shape[dim_idx] == dim_size.getSExtValue()) {
      new_shape.push_back(dim_size.getSExtValue());
    } else if (include_this_dim &&
               src_shape[dim_idx] != dim_size.getSExtValue()) {
      new_shape.push_back(dim_size.getSExtValue());
    }

    dim_idx += 1;
  }

  // create a dynamic gather op
  auto dynamic_gather_op = b.create<mhlo::DynamicGatherOp>(
      op->getLoc(),
      RankedTensorType::get(new_shape, gather_op.getResult()
                                           .getType()
                                           .cast<RankedTensorType>()
                                           .getElementType()),
      op->getOperand(0), op->getOperand(1),
      b.create<tensor::FromElementsOp>(op->getLoc(), slice_sizes_vec)
          .getResult(),
      mhlo::GatherDimensionNumbersAttr::get(
          attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
          attr.getStartIndexMap(), attr.getIndexVectorDim()),
      gather_op.getIndicesAreSorted());
  gather_op.getResult().replaceAllUsesWith(dynamic_gather_op.getResult());

  // Update DynamicGatherOp result shape information
  return propagateHelper<mhlo::DynamicGatherOp>(
      b, dynamic_gather_op.getOperation(), inputCtx);
}

LogicalResult parseInputDynamicDims(
    func::FuncOp main,
    std::vector<std::pair<int, std::vector<int>>>& input_dynamic_dims) {
  auto dict_attr = main->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!dict_attr) {
    return failure();
  }
  if (!dict_attr.get(kDynamicDimsAttr)) {
    return failure();
  }
  StringRef param_str =
      dict_attr.get(kDynamicDimsAttr).dyn_cast<mlir::StringAttr>();

  SmallVector<StringRef, 4> parsed_dynamic_dims;
  param_str.split(parsed_dynamic_dims, "|");
  for (auto kv : parsed_dynamic_dims) {
    SmallVector<StringRef, 4> pair;
    kv.split(pair, ":");
    if (pair.size() != 2) {
      return failure();
    }
    int arg_index = std::stoi(pair[0].str());
    SmallVector<StringRef, 4> dims;
    pair[1].split(dims, ",");
    std::vector<int> dim_vec;
    for (auto dim : dims) {
      dim_vec.push_back(std::stoi(dim.str()));
    }
    input_dynamic_dims.push_back({arg_index, dim_vec});
  }
  return success();
}

void applyShapeContext(ShapeContext& ctx) {
  if (!ctx.value) return;
  auto res_ty = ctx.value.getType().dyn_cast<RankedTensorType>();
  if (!res_ty) return;
  auto elemTy = res_ty.getElementType();
  ctx.value.setType(RankedTensorType::get(ctx.shape, elemTy));
}

std::optional<ShapeContext> propagateOpShape(OpBuilder& rewriter, Operation* op,
                                             ShapeContext& inputCtx) {
  if (isUnaryOp(op)) {
    return ShapeContext(op->getResult(0), inputCtx.shape);
  }
  if (isBinaryOp(op)) {
    return HandleBinaryOp(rewriter, op, inputCtx);
  }
  if (isa<tensor::DimOp>(op)) {
    return propagateHelper<tensor::DimOp>(rewriter, op, inputCtx);
  }
  if (isa<mhlo::BroadcastInDimOp>(op)) {
    return ShapeContext(op->getResult(0), inputCtx.shape);
  }
#define PROPAGATE_OP_HANDLER(opType)                              \
  if (auto t##opType = dyn_cast<mhlo::opType>(op)) {              \
    rewriter.setInsertionPoint(op);                               \
    return propagateHelper<mhlo::opType>(rewriter, op, inputCtx); \
  }
  PROPAGATE_OP_HANDLER(DotOp);
  PROPAGATE_OP_HANDLER(SliceOp);
  PROPAGATE_OP_HANDLER(ReshapeOp);
  PROPAGATE_OP_HANDLER(ConcatenateOp);
  PROPAGATE_OP_HANDLER(ReduceOp);
  PROPAGATE_OP_HANDLER(TransposeOp);
  PROPAGATE_OP_HANDLER(GatherOp);
  PROPAGATE_OP_HANDLER(DynamicGatherOp);
  PROPAGATE_OP_HANDLER(DotGeneralOp);
  PROPAGATE_OP_HANDLER(DynamicReshapeOp);
  PROPAGATE_OP_HANDLER(RealDynamicSliceOp);
  PROPAGATE_OP_HANDLER(DynamicBroadcastInDimOp);
  // PROPAGATE_OP_HANDLER(DimOp);
#undef PROPAGATE_OP_HANDLER
  return std::nullopt;
}  // namespace

bool shouldStopPropagation(Operation* op, ShapeContext& ctx) {
  if (isConcreteShape(ctx)) return true;
  if (isa<func::ReturnOp, mhlo::PadOp, shape::ShapeOfOp, tensor::DimOp>(op))
    return true;
  if (isa<mhlo::ReduceOp>(op->getParentOp())) return true;

  return false;
}

void DiscShapePropagatePass::visitOperator(ModuleOp& m, OpBuilder& rewriter,
                                           Operation* op,
                                           std::stack<ShapeContext>& ctxStack) {
  auto ctx = ctxStack.top();
  if (shouldStopPropagation(op, ctx)) {
    return;
  }
  auto resultShapeCtx = propagateOpShape(rewriter, op, ctx);
  if (!resultShapeCtx) {
    m.emitError("failed propagate shape on op:" +
                op->getName().stripDialect().str());
    signalPassFailure();
    return;
  }
  ctxStack.push(*resultShapeCtx);
  SmallVector<Operation*, 4> ctxUsers(resultShapeCtx->value.getUsers().begin(),
                                      resultShapeCtx->value.getUsers().end());
  for (size_t i = 0; i < ctxUsers.size(); ++i) {
    visitOperator(m, rewriter, ctxUsers[i], ctxStack);
  }
  auto context = ctxStack.top();
  ctxStack.pop();
  applyShapeContext(context);
}

void DiscShapePropagatePass::runOnOperation() {
  ModuleOp m = getOperation();
  auto main = m.lookupSymbol<FuncOp>("main");
  MLIRContext* context = &getContext();
  mlir::OpBuilder rewriter(context);
  OpBuilder b(main);
  if (!main) {
    m.emitError("entry func: main not found");
    signalPassFailure();
    return;
  }
  SmallVector<Type, 4> new_arg_types, new_return_types;
  for (auto arg : main.getArguments()) {
    new_arg_types.push_back(arg.getType());
  }
  // stage1: parse attribute input_dynamic_dims to a map
  std::vector<std::pair<int, std::vector<int>>> input_dynamic_dims;
  if (failed(parseInputDynamicDims(main, input_dynamic_dims))) {
    return;
  }
  // skip this pass if no dynamic dims attribute
  if (input_dynamic_dims.size() == 0) return;
  // stage2: visit all operators to propagate shape
  for (auto pair : input_dynamic_dims) {
    int argIdx = pair.first;
    Value value = main.getArgument(argIdx);
    auto ty = value.getType().cast<RankedTensorType>();
    SmallVector<int64_t> newShape;
    std::copy(ty.getShape().begin(), ty.getShape().end(),
              std::back_inserter(newShape));
    for (auto dim : pair.second) {
      newShape[dim] = ShapedType::kDynamic;
    }
    std::stack<ShapeContext> ctxStack;
    ShapeContext ctx(value, newShape);
    ctxStack.push(ctx);
    auto newType = RankedTensorType::get(newShape, ty.getElementType());
    for (auto user : main.getArgument(argIdx).getUsers()) {
      visitOperator(m, rewriter, user, ctxStack);
    }
    new_arg_types[argIdx] = newType;
    applyShapeContext(ctx);
  }

  // stage3: visit all return operators to update function signature
  main.walk([&](Operation* op) {
    if (isa<func::ReturnOp>(*op)) {
      for (auto operand : op->getOperands()) {
        new_return_types.push_back(operand.getType());
      }
    }
  });
  main.setType(
      FunctionType::get(main.getContext(), new_arg_types, new_return_types));
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapePropagatePass() {
  return std::make_unique<DiscShapePropagatePass>();
}

}  // namespace disc_ral
}  // namespace mlir
