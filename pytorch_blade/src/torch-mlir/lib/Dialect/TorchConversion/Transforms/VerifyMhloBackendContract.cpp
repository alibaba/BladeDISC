//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h> // from tf repo
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToStd/TorchToStd.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
class VerifyMhloBackendContractPass
    : public VerifyMhloBackendContractBase<VerifyMhloBackendContractPass> {

  template <typename T> Type removeUser(Value arg, Type default_type) {
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

  template <typename T> Value backtraceOperand(Value operand) {
    auto op = operand.getDefiningOp();
    if (mlir::isa<T>(op)) {
      return op->getOperand(0);
    }
    return operand;
  }

  void refineFuncArgTypes(func::FuncOp &func) {
    SmallVector<Type> new_types;
    SmallVector<Value> old_args;

    auto n_args = func.getNumArguments();
    auto &body = func.getBody();
    for (size_t idx = 0; idx < n_args; ++idx) {
      Value arg = body.getArgument(idx);
      Type new_type = arg.getType();
      old_args.push_back(arg);
      new_type = removeUser<TensorStaticInfoCastOp>(arg, arg.getType());
      new_type = removeUser<CopyToValueTensorOp>(arg, new_type);
      new_type = removeUser<ToBuiltinTensorOp>(arg, new_type);
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
      return signalPassFailure();
    }

    auto funcType = func.getFunctionType();
    ConversionPatternRewriter rewriter(funcType.getContext());
    rewriter.updateRootInPlace(returnOp, [&] {
      auto n_operands = returnOp->getNumOperands();
      for (size_t k = 0; k < n_operands; ++k) {
        auto operand = returnOp->getOperand(k);
        operand = backtraceOperand<ToBuiltinTensorOp>(operand);
        operand = backtraceOperand<FromBuiltinTensorOp>(operand);
        returnOp->setOperand(k, operand);
      }
    });

    // Update the function type.
    func.setType(FunctionType::get(funcType.getContext(), new_types,
                                   returnOp->getOperandTypes()));
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    module.walk([&](func::FuncOp funcOp) { refineFuncArgTypes(funcOp); });

    ::mlir::PassManager pm(context, ::mlir::OpPassManager::Nesting::Implicit);
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    module.dump();
    if (failed(pm.run(module))) {
      emitError(module.getLoc())
          << "Module does not conform to the MHLO backend contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }
    module.dump();

    TypeConverter converter;
    converter.addConversion([](TensorType type) -> Type {
      if (BaseMemRefType::isValidElementType(type.getElementType()))
        return type;
      return nullptr;
    });

    auto opHasLegalTypes = [&](Operation *op) { return converter.isLegal(op); };

    ConversionTarget target(*context);

    // Structural operations.
    target.addDynamicallyLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>(
        opHasLegalTypes);
    // Basic scalar operations.
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<chlo::ChloDialect>();
    target.addIllegalOp<TensorStaticInfoCastOp>();
    target.addIllegalOp<CopyToValueTensorOp>();
    target.addIllegalOp<ToBuiltinTensorOp>();

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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createVerifyMhloBackendContractPass() {
  return std::make_unique<VerifyMhloBackendContractPass>();
}


void TorchConversion::createTorchBackendToMhloBackendPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  // Check some invariants to catch errors in a clear way.
  pm.addPass(
      TorchConversion::createVerifyInvariantsBeforeBackendLoweringPass());

  pm.addNestedPass<func::FuncOp>(createConvertTorchToMhloPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStdPass());

  // Perform rank broadcasting so MhloToLinalg pass works
  // pm.addNestedPass<func::FuncOp>(createMhloMakeBroadcastablePass());

  if (options.optimize) {
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Finish the type conversion from `torch` types to the types of the
  // MHLO backend contract.
  // pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  // pm.addNestedPass<func::FuncOp>(
  //     TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that MHLO backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyMhloBackendContractPass());
}

