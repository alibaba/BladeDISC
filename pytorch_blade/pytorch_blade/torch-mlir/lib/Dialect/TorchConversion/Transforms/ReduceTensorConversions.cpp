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

#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
template <typename T>
Type removeUser(Value arg, Type default_type) {
  Type new_type = default_type;
  for (auto user : arg.getUsers()) {
    if (mlir::isa<T>(user)) {
      auto val = user->getResult(0);
      new_type = val.getType();
      val.replaceAllUsesWith(arg);
      user->erase();
    }
  }
  return new_type;
}

template <typename T>
Value backtraceOperand(Value operand) {
  auto op = operand.getDefiningOp();
  if (op && mlir::isa<T>(op)) {
    return op->getOperand(0);
  }
  return operand;
}
} // namespace

// This conversion will remove tensor conversion operators from
// the func::FuncOp.
LogicalResult Torch::reduceTensorConversions(func::FuncOp& func) {
  SmallVector<Type> new_types;
  SmallVector<Value> old_args;

  auto n_args = func.getNumArguments();
  auto& body = func.getBody();
  for (size_t idx = 0; idx < n_args; ++idx) {
    Value arg = body.getArgument(idx);
    Type new_type = arg.getType();
    old_args.push_back(arg);
    new_type = removeUser<TensorStaticInfoCastOp>(arg, arg.getType());
    new_type = removeUser<CopyToValueTensorOp>(arg, new_type);
    new_type = removeUser<ToBuiltinTensorOp>(arg, new_type);
    new_type = removeUser<ToF64Op>(arg, new_type);
    new_type = removeUser<ToI64Op>(arg, new_type);
    new_types.push_back(new_type);
  }

  for (size_t idx = 0; idx < n_args; ++idx) {
    auto old_arg = old_args[idx];
    auto new_arg = body.addArgument(new_types[idx], old_arg.getLoc());
    old_arg.replaceAllUsesWith(new_arg);
    body.eraseArgument(0);
  }
  // Find the unique return op.
  func::ReturnOp returnOp;
  WalkResult walkResult = func.walk([&](func::ReturnOp op) {
    if (returnOp)
      return WalkResult::interrupt();
    returnOp = op;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    func.emitError() << "unimplemented: refining returns for function with "
                        "more than one return op";
    return failure();
  }

  auto funcType = func.getFunctionType();
  ConversionPatternRewriter rewriter(funcType.getContext());
  rewriter.updateRootInPlace(returnOp, [&] {
    auto n_operands = returnOp->getNumOperands();
    for (size_t k = 0; k < n_operands; ++k) {
      auto operand = returnOp->getOperand(k);
      operand = backtraceOperand<ToBuiltinTensorOp>(operand);
      operand = backtraceOperand<TensorStaticInfoCastOp>(operand);
      operand = backtraceOperand<CopyToValueTensorOp>(operand);
      operand = backtraceOperand<CopyToNonValueTensorOp>(operand);
      operand = backtraceOperand<FromBuiltinTensorOp>(operand);
      operand = backtraceOperand<FromF64Op>(operand);
      operand = backtraceOperand<FromI64Op>(operand);
      returnOp->setOperand(k, operand);
    }
  });

  // Update the function type.
  func.setType(FunctionType::get(
      funcType.getContext(), new_types, returnOp->getOperandTypes()));
  return success();
}
