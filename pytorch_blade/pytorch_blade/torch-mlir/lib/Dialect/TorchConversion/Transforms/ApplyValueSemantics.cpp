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

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
static Type getContainerOrTensorTypeWithValueSemantics(Type type) {
  if (auto optionalType = type.dyn_cast<OptionalType>()) {
    Type newContainedType = getContainerOrTensorTypeWithValueSemantics(
        optionalType.getContainedType());
    if (newContainedType)
      return OptionalType::get(newContainedType);
  } else if (auto listType = type.dyn_cast<ListType>()) {
    Type newContainedType =
        getContainerOrTensorTypeWithValueSemantics(listType.getContainedType());
    if (newContainedType)
      return ListType::get(newContainedType);
  } else if (auto tensorType = type.dyn_cast<NonValueTensorType>()) {
    return tensorType.getWithValueSemantics();
  }
  return nullptr;
}

//  Greedily apply value-semantic tensors where possible to make the program
//  more easier to convert by later passes (also, backends prefer value
//  semantics as well).
class ApplyValueSemanticsPass
    : public TorchConversion::ApplyValueSemanticsBase<ApplyValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    mlir::OpBuilder rewriter(context);
    auto func = getOperation();

    SmallVector<Operation*> deadOps;
    // 1. Replace all return type from "!torch.tensor" to "!torch.vtensor"
    // 2. Erase all "torch.copy.to_tensor" and "torch.copy.to_vtensor"
    func.walk([&](Operation* op) {
      if (isa<CopyToNonValueTensorOp, CopyToValueTensorOp>(op)) {
        op->getResult(0).replaceAllUsesWith(op->getOpOperand(0).get());
        deadOps.push_back(op);
        return;
      }

      for (auto val : op->getResults()) {
        if (auto newType =
                getContainerOrTensorTypeWithValueSemantics(val.getType()))
          val.setType(newType);
      }
    });
    for (auto op : deadOps) {
      op->erase();
    }
    reduceTensorConversions(func);

    deadOps.clear();
    func.walk([&](mlir::Operation* op) {
      if (isa<TensorStaticInfoCastOp>(op)) {
        op->getResult(0).replaceAllUsesWith(op->getOpOperand(0).get());
        deadOps.push_back(op);
        return;
      }
    });
    for (auto op : deadOps) {
      op->erase();
    }

    // fix NonValueTensor type operator
    func.walk([&](Operation* op) {
      if (isa<OverwriteTensorContentsOp>(op)) {
        rewriter.setInsertionPoint(op);
        auto nonValueTensor = rewriter.create<CopyToNonValueTensorOp>(
            op->getLoc(), op->getOperands()[1]);
        op->setOperand(1, nonValueTensor);
      }
    });

    func.walk([&](Operation* op) {
      for (auto val : op->getResults()) {
        auto valTy = val.getType().dyn_cast_or_null<ValueTensorType>();
        if (!valTy) {
          continue;
        }
        if (!valTy.hasDtype() || !valTy.hasSizes()) {
          op->emitError("output has incomplete tensor type and sizes");
          signalPassFailure();
        }
      }
    });

    ConversionTarget target(*context);
    target.addLegalDialect<
        Torch::TorchDialect,
        func::FuncDialect,
        arith::ArithDialect,
        tensor::TensorDialect,
        cf::ControlFlowDialect,
        math::MathDialect>();
    target.addLegalOp<ModuleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      // We avoid `func.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      emitError(func.getLoc())
          << "Func does not conform to the MHLO backend contract. "
             "See dialect conversion legality information above.";
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createApplyValueSemanticsPass() {
  return std::make_unique<ApplyValueSemanticsPass>();
}
