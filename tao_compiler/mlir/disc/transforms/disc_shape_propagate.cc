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
struct DiscShapePropagatePass
    : public DiscShapePropagatePassBase<DiscShapePropagatePass> {
  DiscShapePropagatePass()
      : DiscShapePropagatePassBase<
            DiscShapePropagatePass>::DiscShapePropagatePassBase() {}

  void runOnOperation() override;
};

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

LogicalResult HandleBinaryOp(OpBuilder& b, Operation* op) {
  // set result type same as input type
  auto shape = op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto elemTy =
      op->getOperand(0).getType().cast<RankedTensorType>().getElementType();
  auto resultElemTy =
      op->getResult(0).getType().cast<RankedTensorType>().getElementType();
  op->getResult(0).setType(RankedTensorType::get(shape, resultElemTy));
  if (auto const_op =
          dyn_cast<mhlo::ConstantOp>(op->getOperand(1).getDefiningOp())) {
    b.setInsertionPoint(op);
    auto dense_attr = const_op.getValue().dyn_cast<mlir::DenseElementsAttr>();

    int64_t value = (*dense_attr.getValues<APInt>().begin()).getSExtValue();
    auto scalar_const_op = getConstTensor(b, op, {value}, {});
    Value inputShape =
        b.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0));
    auto rank = shape.size();
    SmallVector<int64_t> boradcast_dim;
    boradcast_dim.push_back(static_cast<int64_t>(rank));

    auto bcast_op = b.create<mhlo::DynamicBroadcastInDimOp>(
        op->getLoc(), RankedTensorType::get(shape, elemTy),
        scalar_const_op.value(), inputShape, b.getI64TensorAttr({}));
    const_op.getResult().replaceAllUsesWith(bcast_op.getResult());
    const_op.erase();
  }
  return success();
}

LogicalResult HandleUnaryOp(OpBuilder& b, Operation* op) {
  int n = op->getOperands().size();
  // set result type same as input type
  auto shape = op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto elemTy =
      op->getOperand(0).getType().cast<RankedTensorType>().getElementType();
  auto resultElemTy =
      op->getResult(0).getType().cast<RankedTensorType>().getElementType();
  op->getResult(0).setType(RankedTensorType::get(shape, resultElemTy));
  return success();
}
LogicalResult HandleDot(OpBuilder& b, Operation* op) {
  if (!isa<mhlo::DotOp>(op)) return success();
  auto dot_op = cast<mhlo::DotOp>(op);
  auto lhs_shape =
      dot_op.getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto rhs_shape =
      dot_op.getOperand(1).getType().cast<RankedTensorType>().getShape();
  auto result_shape =
      dot_op.getResult().getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t, 4> new_shape;
  new_shape.push_back(lhs_shape[0]);
  new_shape.push_back(rhs_shape[1]);
  dot_op.getResult().setType(RankedTensorType::get(
      new_shape,
      dot_op.getResult().getType().cast<RankedTensorType>().getElementType()));
  return success();
}

bool isUnaryOp(Operation* op) {
  return isa<mhlo::AddOp>(*op) || isa<mhlo::CompareOp>(*op) ||
         isa<mhlo::SelectOp>(*op) || isa<mhlo::ConvertOp>(*op);
}

bool isBinaryOp(Operation* op) { return isa<mhlo::ConvertOp>(op); }

LogicalResult eraseInputTensorShapesFromAttr(
    FuncOp& main, StringRef input_dynamic_dims_attr,
    SmallVector<Type, 4>& new_arg_types) {
  SmallVector<StringRef, 4> parsed_dynamic_dims;
  input_dynamic_dims_attr.split(parsed_dynamic_dims, "|");
  for (auto kv : parsed_dynamic_dims) {
    SmallVector<StringRef, 4> pair;
    kv.split(pair, ":");
    if (pair.size() != 2) {
      main.emitError("input_dynamic_dims format error");
      return failure();
    }
    int arg_index = std::stoi(pair[0].str());
    auto arg = main.getArgument(arg_index);

    auto arg_ty = new_arg_types[arg_index].dyn_cast<RankedTensorType>();
    if (!arg_ty) {
      main.emitError("input tensor type not found");
      return failure();
    }
    auto shape = arg_ty.getShape();
    SmallVector<int64_t, 4> new_shape(shape.begin(), shape.end());
    SmallVector<StringRef, 4> dims;
    pair[1].split(dims, ",");
    for (auto dim : dims) {
      new_shape[std::stoi(dim.str())] = ShapedType::kDynamic;
    }
    auto new_ty = RankedTensorType::get(new_shape, arg_ty.getElementType());
    arg.setType(new_ty);
    new_arg_types[arg_index] = new_ty;
    main.getArgument(arg_index).setType(new_ty);
  }
  return success();
}
void DiscShapePropagatePass::runOnOperation() {
  ModuleOp m = getOperation();
  FuncOp main = m.lookupSymbol<FuncOp>("main");
  MLIRContext* context = &getContext();
  mlir::OpBuilder rewriter(context);
  OpBuilder b(main);
  if (!main) {
    m.emitError("entry func: main not found");
    signalPassFailure();
    return;
  }

  // stage1: fixup input tensor shapes accroding to attr
  auto dict_attr = main->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!dict_attr) {
    signalPassFailure();
    return;
  }
  auto param_str =
      dict_attr.get("input_dynamic_dims").dyn_cast<mlir::StringAttr>();
  if (param_str.getValue().empty()) {
    return;
  }
  SmallVector<Type, 4> new_arg_types;
  for (auto arg : main.getArguments()) {
    new_arg_types.push_back(arg.getType());
  }
  if (failed(eraseInputTensorShapesFromAttr(main, param_str, new_arg_types))) {
    m.emitError("failed erase input tensor shapes from attr");
    signalPassFailure();
    return;
  }

  // stage2: propagate tensor shapes or rewrite graph with dynamic operator
  SmallVector<Type, 4> new_return_types;
  main.walk([&](Operation* op) {
    if (isUnaryOp(op)) {
      if (failed(HandleUnaryOp(rewriter, op))) {
        m.emitError("failed handle unary op: ");
        signalPassFailure();
        return;
      }
    }
    if (failed(HandleDot(rewriter, op))) {
      m.emitError("failed handle dot op: ");
      signalPassFailure();
      return;
    }
    if (isa<func::ReturnOp>(*op)) {
      for (auto operand : op->getOperands()) {
        new_return_types.push_back(operand.getType());
      }
    }
  });
  // stage3: update function signature
  main.setType(
      FunctionType::get(main.getContext(), new_arg_types, new_return_types));
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapePropagatePass() {
  return std::make_unique<DiscShapePropagatePass>();
}

}  // namespace disc_ral
}  // namespace mlir
