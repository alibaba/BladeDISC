//===-- MhloPasses.td - Pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_MHLO_PASSES
#define TORCHMLIR_CONVERSION_MHLO_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTorchToMhlo: Pass<"convert-torch-to-mhlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to MHLO ops";
  let description = [{
    This pass assumes that MHLO ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::torch::createConvertTorchToMhloPass()";
}

 
def VerifyMhloBackendContract : Pass<"torch-verify-mhlo-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the mhlo backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyMhloBackendContractPass()";
}

#endif // TORCHMLIR_CONVERSION_MHLO_PASSES
