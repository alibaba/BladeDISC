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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mhlo/IR/hlo_ops.h" // from mhlo

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "stablehlo/dialect/ChloOps.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
class VerifyMhloBackendContractPass
    : public VerifyMhloBackendContractBase<VerifyMhloBackendContractPass> {
  void updateFuncType(func::FuncOp func) {
    func::ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](func::ReturnOp op) {
      if (returnOp)
        return WalkResult::interrupt();
      returnOp = op;
      return WalkResult::advance();
    });

    auto context = func.getContext();
    // the convert_ty function will convert !torch.vtensor to builtin tensor
    auto convert_ty = [&](BlockArgument& arg) -> Type {
      Type type = arg.getType();
      if (type.isa<BaseTensorType>()) {
        auto vtensorTy = type.dyn_cast_or_null<ValueTensorType>();
        if (vtensorTy) {
          type = vtensorTy.toBuiltinTensor();
        } else {
          auto tensorTy = type.dyn_cast<NonValueTensorType>();
          type = tensorTy.getWithValueSemantics().toBuiltinTensor();
        }
        arg.setType(type);
      } else if (type.isa<Torch::FloatType>()) {
        type = Float64Type::get(context);
        arg.setType(type);
      } else if (type.isa<Torch::IntType>()) {
        type = IntegerType::get(context, 64);
        arg.setType(type);
      }
      return type;
    };

    SmallVector<Type> new_arg_types;
    auto n_args = func.getNumArguments();
    new_arg_types.reserve(n_args);

    auto& body = func.getBody();
    for (auto arg : body.getArguments()) {
      new_arg_types.push_back(convert_ty(arg));
    }
    auto ret_types = returnOp.getOperandTypes();
    // update the argumetns type
    func.setType(FunctionType::get(context, new_arg_types, ret_types));
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto module = getOperation();
    SmallVector<mlir::Operation*> toDestroy;
    module.walk([&](func::FuncOp func) {
      (void)reduceTensorConversions(func);
      // After the refinement all used arguments are changed to builtin tensor.
      // There exists some arguments have no users, but should be update.
      // Such as the following `%arg1`:
      // ```
      //   func @main(%arg0: tensor<2x3x224x224xf32>, %arg1:
      //   !torch.vtensor<[6,224],f16>)
      //         -> tensor<2x3x224x224xf16> {
      //     %0 = mhlo.convert(%arg0) :
      //          (tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf16>
      //     return %0 : tensor<2x3x224x224xf16>
      //   }
      // ```
      updateFuncType(func);

      // 1. Erase all "torch_c.from_i64" and "torch_c.to_i64"
      func.walk([&](mlir::Operation* op) {
        if (isa<ToI64Op, FromI64Op, ToF64Op, FromF64Op>(op)) {
          op->getResult(0).replaceAllUsesWith(op->getOpOperand(0).get());
          toDestroy.push_back(op);
        }
        if (isa<cf::AssertOp>(op)) {
          toDestroy.push_back(op);
        }
      });
    });
    for (auto op : toDestroy) {
      op->erase();
    }

    ::mlir::PassManager pm(
        context,
        module.getOperationName(),
        ::mlir::OpPassManager::Nesting::Implicit);
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    if (failed(pm.run(module))) {
      emitError(module.getLoc())
          << "Module does not conform to the MHLO backend contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }

    TypeConverter converter;
    converter.addConversion([](TensorType type) -> Type {
      if (BaseMemRefType::isValidElementType(type.getElementType()))
        return type;
      return nullptr;
    });

    auto opHasLegalTypes = [&](Operation* op) { return converter.isLegal(op); };

    ConversionTarget target(*context);
    // Structural operations.
    target.addDynamicallyLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>(
        opHasLegalTypes);
    target.addLegalDialect<
        arith::ArithDialect,
        chlo::ChloDialect,
        mhlo::MhloDialect,
        mhlo_disc::MhloDiscDialect,
        tensor::TensorDialect>();
    RewritePatternSet patterns(context);
    populateReturnOpTypeConversionPattern(patterns, converter);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      // We avoid `module.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      emitError(module.getLoc())
          << "Module does not conform to the MHLO backend contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::torch::TorchConversion::
    createVerifyMhloBackendContractPass() {
  return std::make_unique<VerifyMhloBackendContractPass>();
}
