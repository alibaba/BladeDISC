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
//#include "mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/MhloPasses.h.inc"
} // end namespace

void ::mlir::torch::registerTorchToMhloPasses() {
  ::registerPasses();
  ::mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "torch-backend-to-mhlo-backend-pipeline",
      "Pipeline lowering torch backend contract to mhlo backend "
      "contract.",
      ::mlir::torch::TorchConversion::createTorchBackendToMhloBackendPipeline);
}