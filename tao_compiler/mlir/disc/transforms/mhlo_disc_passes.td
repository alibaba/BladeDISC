/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def DiscHloLegalizeToLhloPass : Pass<"disc-hlo-legalize-to-lhlo", "ModuleOp"> {
  let summary = "convert mhlo_disc ops to lmhlo_disc ops.";
  let constructor = "createDiscLegalizeToLhloPass()";
}

def DiscLhloRewriterPass: Pass<"disc-lhlo-rewriter", "ModuleOp"> {
  let summary = "rewrite lmhlo ops to lmhlo_disc ops.";
  let constructor = "createDiscLhloRewriterPass()";
}

def DiscOptimizationBarrierExpandPass : Pass<"disc-optimization-barrier-expand", "ModuleOp"> {
  let summary = "Expand OptimizationBarrierOp";
  let constructor = "createDiscOptimizationBarrierExpandPass()";
}

def DiscArgsMutationExpandPass : Pass<"disc-argsmutation-expand", "ModuleOp"> {
  let summary = "Expand ArgsMutationOp";
  let constructor = "createDiscArgsMutationExpandPass()";
}

def DiscOpSchedulePass : Pass<"disc-op-schedule", "ModuleOp"> {
  let summary = "Schedule ops in a function";
  let constructor = "createDiscOpSchedulePass()";
}
