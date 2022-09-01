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
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/MhloPasses.h.inc"
} // end namespace

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

void ::mlir::torch::registerTorchToMhloPasses() {
  ::registerPasses();
  ::mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-backend-to-mhlo-backend-pipeline",
      "Pipeline lowering torch backend contract to mhlo backend "
      "contract.",
      mlir::torch::createDiscTorchBackendToMhloBackendPipeline);
}

void mlir::torch::createDiscTorchBackendToMhloBackendPipeline(
    OpPassManager& pm,
    const Torch::TorchLoweringPipelineOptions& options) {
  ::mlir::torch::Torch::TorchLoweringPipelineOptions funcOptions;
  funcOptions.decompose = false;
  ::mlir::torch::createDiscTorchFunctionToTorchBackendPipeline(pm, funcOptions);

  // Add decompose passes
  pm.addNestedPass<func::FuncOp>(createApplyValueSemanticsPass());
  pm.addNestedPass<func::FuncOp>(createDiscDecomposeComplexOpsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      Torch::createDecomposeComplexOpsPass(/*legalOps*/ {}));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Do mhlo lowering
  // pm.addNestedPass<func::FuncOp>(createApplyValueSemanticsPass());
  pm.addNestedPass<func::FuncOp>(createDiscConvertTorchToMhloPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToMhloPass(
      /*enableStaticShape*/ false, /*enableI32Index*/ true));
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

  // Perform rank broadcasting so MhloToLinalg pass works
  // pm.addNestedPass<func::FuncOp>(createMhloMakeBroadcastablePass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Verify that we have lowered to the form that MHLO backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(createVerifyMhloBackendContractPass());
}

void mlir::torch::createDiscTorchFunctionToTorchBackendPipeline(
    OpPassManager& pm,
    const TorchLoweringPipelineOptions& options) {
  // General considerations: As a matter of bring-up, we are simultaneously
  // building out the frontend pipeline and also co-developing the backend
  // support story as well. This means that sometimes the most expedient way to
  // support a given program is to "optimize hard enough" that the parts of the
  // program that touch unimplemented backend support go away (constant folded,
  // dead-code-eliminated, etc.). In the fullness of time, most of that
  // optimization should not be necessary, and we should have an "O0" pipeline
  // that runs practically no optimizations.
  // However, as a matter of expediency, at the moment we do run those
  // optimizations. We guard those passes under the `options.optimize` option
  // (which default to true, currently). We leave notes with the `OPT-ONLY` tag
  // why we currently need that pass for correctness.
  // We should eventually remove those passes from the default pipeline once
  // backends have enough support.
  // In particular the following features are needed in some form from backends:
  // - Error handling (RaiseException + error string formatting)
  // - First-class list type
  // - torch.global_slot lowering
  // - ...
  // Please try to keep this list somewhat up to date when adding
  // "optimize hard enough that it works" transformations.

  // Incorporate user annotations and remove signature Python-isms.
  pm.addPass(createAdjustCallingConventionsPass());

  // Eliminate the PrimTupleIndexOp generated from the
  // adjustCallingConventions
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Reduce variants of ops to a smaller set of primitives.
  pm.addNestedPass<func::FuncOp>(createReduceOpVariantsPass());

  //===--------------------------------------------------------------------===//
  // Lowering to ranked !torch.vtensors of known dtype.
  //===--------------------------------------------------------------------===//

  // Convert the bulk of non-ABI-visible !torch.tensor's to !torch.vtensor's.
  pm.addNestedPass<func::FuncOp>(Torch::createMaximizeValueSemanticsPass());

  // Do shape refinement.
  // This must be run before RefineTypes (which primarily does dtype inference),
  // because Torch type promotion rules actually depend on the shape of the
  // operand.
  // createTorchShapeRefinementPipeline(pm, options);
  // Refine types in the program, which mainly means inferring dtypes of ops.
  // pm.addNestedPass<func::FuncOp>(Torch::createRefineTypesPass());

  // Propagate to ABI return types the shape/dtype information discovered by
  // the previous pass. Doing this is ABI-compatible for our backends.
  pm.addPass(Torch::createRefinePublicReturnPass());

  // This can fold away some branches given the information got from
  // RefineTypes before doing maximize value sematics which only works with
  // basic blocks.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}
