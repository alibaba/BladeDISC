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
#include "utils/env.h"

namespace impl {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/MhloPasses.h.inc"
} // end namespace impl

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

void ::mlir::torch::registerTorchToMhloPasses() {
  ::impl::registerPasses();
  ::mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-backend-to-mhlo-backend-pipeline",
      "Pipeline lowering torch backend contract to mhlo backend "
      "contract.",
      mlir::torch::createDiscTorchBackendToMhloBackendPipeline);
}

void mlir::torch::createDiscTorchBackendToMhloBackendPipeline(
    OpPassManager& pm,
    const Torch::TorchLoweringPipelineOptions& options) {
  pm.addNestedPass<func::FuncOp>(createDiscUpgradeLegacyOpsPass());
  ::mlir::torch::Torch::TorchLoweringPipelineOptions funcOptions;
  funcOptions.decompose = false;
  ::mlir::torch::createDiscTorchFunctionToTorchBackendPipeline(pm, funcOptions);
  // Apply pdl pattern match
  std::string disc_torch_pdl_files;
  std::string disc_torch_pdll_include_dirs;
  disc_torch_pdl_files =
      mlir::torch::utils::env::ReadStringFromEnvVar("DISC_TORCH_PDL_FILES", "");
  disc_torch_pdll_include_dirs = mlir::torch::utils::env::ReadStringFromEnvVar(
      "DISC_TORCH_PDLL_INCLUDE_DIRS", "");
  pm.addNestedPass<func::FuncOp>(createApplyDiscPdlPatternsPass(
      disc_torch_pdl_files, disc_torch_pdll_include_dirs));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Add decompose passes
  pm.addNestedPass<func::FuncOp>(createDiscDecomposeComplexOpsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Add simplify patterns pass
  pm.addNestedPass<func::FuncOp>(createDiscSimplifyPatternsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(Torch::createDecomposeComplexOpsPass(
      /*legalOps*/ {"torch.aten.slice_scatter"}));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createApplyDiscPdlPatternsPass(
      disc_torch_pdl_files, disc_torch_pdll_include_dirs));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // TorchMLIR DecomposeComplexOpsPass might generate new operators
  // that need to be decomposed further before DISC passes
  pm.addNestedPass<func::FuncOp>(createDiscDecomposeComplexOpsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Do mhlo lowering
  pm.addNestedPass<func::FuncOp>(createDiscConvertTorchToMhloPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToMhloPass(
      /*enableStaticShape*/ false, /*enableI32Index*/ true));
  pm.addNestedPass<func::FuncOp>(createDiscConvertTorchToDiscMhlo());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

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
  // Reduce variants of ops to a smaller set of primitives.
  pm.addNestedPass<func::FuncOp>(createReduceOpVariantsPass());

  //===--------------------------------------------------------------------===//
  // Lowering to ranked !torch.vtensors of known dtype.
  //===--------------------------------------------------------------------===//

  // Convert the bulk of non-ABI-visible !torch.tensor's to !torch.vtensor's.
  pm.addNestedPass<func::FuncOp>(createApplyValueSemanticsPass());

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
