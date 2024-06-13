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
  void runOnOperation() override;
};
bool isBinaryOp(Operation* op) {
  return isa<mhlo::AddOp>(*op) || isa<mhlo::CompareOp>(*op) ||
         isa<mhlo::SelectOp>(*op) || isa<mhlo::ConvertOp>(*op);
}

bool isUnaryOp(Operation* op) {
  return isa<mhlo::ConvertOp, mhlo::ScatterOp>(op);
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
  if (!isBinaryOp(op)) return std::nullopt;
  if (op->getOperand(1).isa<BlockArgument>()) {
    return ShapeContext(op->getResult(0), inputCtx.shape);
  }
  if (auto const_op =
          dyn_cast<mhlo::ConstantOp>(op->getOperand(1).getDefiningOp())) {
    auto elemTy =
        op->getOperand(0).getType().cast<RankedTensorType>().getElementType();
    b.setInsertionPoint(op);
    auto dense_attr = const_op.getValue().dyn_cast<mlir::DenseElementsAttr>();
    int64_t value = (*dense_attr.getValues<APInt>().begin()).getSExtValue();
    auto scalar_const_op = getConstTensor(b, op, {value}, {});
    Value inputShape =
        b.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0));
    auto rank = inputCtx.shape.size();
    SmallVector<int64_t> boradcast_dim;
    boradcast_dim.push_back(static_cast<int64_t>(rank));

    auto bcast_op = b.create<mhlo::DynamicBroadcastInDimOp>(
        op->getLoc(), RankedTensorType::get(inputCtx.shape, elemTy),
        scalar_const_op.value(), inputShape, b.getI64TensorAttr({}));
    const_op.getResult().replaceAllUsesWith(bcast_op.getResult());
    const_op.erase();
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
  auto dot_op = dyn_cast_or_null<mhlo::DotOp>(op);
  if (!dot_op) return std::nullopt;

  auto lhs_shape =
      dot_op.getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto rhs_shape =
      dot_op.getOperand(1).getType().cast<RankedTensorType>().getShape();
  auto result_shape =
      dot_op.getResult().getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> new_shape;
  new_shape.push_back(lhs_shape[0]);
  new_shape.push_back(rhs_shape[1]);
  return ShapeContext(op->getResult(0), new_shape);
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
      new_shape.push_back(ShapedType::kDynamic);
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
  if (auto ctx = HandleBinaryOp(rewriter, op, inputCtx)) {
    return ctx;
  }
  using PropagationFunc =
      std::optional<ShapeContext> (*)(OpBuilder&, Operation*, ShapeContext&);
  const std::vector<PropagationFunc> propagationFunctions = {
      propagateHelper<mhlo::ConcatenateOp>,
      propagateHelper<mhlo::DotOp>,
      propagateHelper<mhlo::DynamicGatherOp>,
      propagateHelper<mhlo::GatherOp>,
      propagateHelper<mhlo::ReduceOp>,
      propagateHelper<mhlo::TransposeOp>,
      propagateHelper<tensor::DimOp>,
  };
  // Iterate over the propagation functions and apply each one
  for (const auto& propagate : propagationFunctions) {
    if (auto ctx = propagate(rewriter, op, inputCtx)) {
      return ctx;
    }
  }
  return std::nullopt;
}

void visitOperator(ModuleOp& m, OpBuilder& rewriter, Operation* op,
                   ShapeContext& ctx) {
  if (isConcreteShape(ctx)) return;
  // later to process return operators
  if (isa<func::ReturnOp>(op)) return;

  auto resultShapeCtx = propagateOpShape(rewriter, op, ctx);
  if (!resultShapeCtx) {
    m.emitError("failed update shape context on op:" +
                op->getName().stripDialect().str());
    return;
  }

  for (auto user : op->getResult(0).getUsers()) {
    visitOperator(m, rewriter, user, resultShapeCtx.value());
  }
  applyShapeContext(*resultShapeCtx);
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
    ShapeContext ctx(value, newShape);
    auto newType = RankedTensorType::get(newShape, ty.getElementType());
    for (auto user : main.getArgument(argIdx).getUsers()) {
      visitOperator(m, rewriter, user, ctx);
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
