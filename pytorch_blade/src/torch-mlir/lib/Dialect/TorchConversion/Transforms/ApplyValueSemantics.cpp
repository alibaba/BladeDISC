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

//===- ApplyValueSemantics.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ApplyValueSemanticsPass
    : public TorchConversion::ApplyValueSemanticsBase<ApplyValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto func = getOperation();

    func.walk([&](mlir::Operation* op) {
      if (isa<CopyToNonValueTensorOp, CopyToValueTensorOp>(op)) {
        op->getResult(0).replaceAllUsesWith(op->getOpOperand(0).get());
        op->erase();
        return;
      }

      // process Operation `op`
      for (auto val : op->getResults()) {
        auto tensor_type = val.getType().dyn_cast_or_null<NonValueTensorType>();
        if (!tensor_type) {
          continue;
        }
        val.setType(tensor_type.getWithValueSemantics());
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createApplyValueSemanticsPass() {
  return std::make_unique<ApplyValueSemanticsPass>();
}
