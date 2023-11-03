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

//===- torch-mlir-opt.cpp - MLIR Optimizer Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/InitAll.h"

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"
using namespace mlir;

int main(int argc, char** argv) {
  mlir::torch::registerTorchToMhloPasses();
  registerAllPasses();
  mlir::torch::registerAllPasses();
  mlir::mhlo::registerStablehloLegalizeToHloPass();

  DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
