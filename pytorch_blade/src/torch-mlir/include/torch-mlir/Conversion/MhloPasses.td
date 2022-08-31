//===-- MhloPasses.td - Pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_DISC_MHLO_PASSES
#define TORCHMLIR_CONVERSION_DISC_MHLO_PASSES

include "mlir/Pass/PassBase.td"


def DiscConvertTorchToMhlo: Pass<"convert-disc-torch-to-mhlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to MHLO ops(Using by DISC)";
  let description = [{
    This pass assumes that MHLO ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::torch::TorchConversion::createDiscConvertTorchToMhloPass()";
}

def ApplyValueSemantics : Pass<"torch-apply-value-semantics", "func::FuncOp"> {
  let summary = "Use value-semantic tensors where possible.";
  let description = [{
    Use value-semantic tensors where possible to make the program more
    analyzable by later passes (also, backends prefer value semantics as well).

    This pass is analogous to an SSA-formation pass in a
    traditional compiler, with the added complication that arrays can alias
    each other in interesting ways.

    The current code doesn't implement any fancy algorithm, and is intended
    to be just sufficient for a first e2e spike. An algorithm inspired by the
    SSA formation literature will need to be implemented.

    Also, this pass doesn't currently handle interprocedural rewriting
    (of private functions), which is even more complex.
  }];
  let constructor = "mlir::torch::TorchConversion::createApplyValueSemanticsPass()";
}

def VerifyMhloBackendContract : Pass<"torch-verify-mhlo-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the mhlo backend contract";
  let constructor = "mlir::torch::TorchConversion::createVerifyMhloBackendContractPass()";
}

def DiscDecomposeComplexOps : Pass<"torch-disc-decompose-complex-ops", "func::FuncOp"> {
  let summary = "Decompose Torch complex operators";
  let description = [{
    Decompose torch operation that are losslessly represented as combinations of
    other operations, modulo appropropriate compiler fusion. Note that this pass
    is similar in spirit to ReduceOpVariants, but ReduceOpVariants is about
    systematic reductions of a large number of ops at once, guided mostly by
    traits.

    An example of the transformations done in this pass is:
    - convert aten.softmax to softmax(x, dim)
            => tmp=exp(x); tmp / sum(tmp, dim, keepdim=True)
  }];
  let constructor = "mlir::torch::TorchConversion::createDiscDecomposeComplexOpsPass()";
}

#endif // TORCHMLIR_CONVERSION_DISC_MHLO_PASSES
