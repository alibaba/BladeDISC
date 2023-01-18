/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

def DiscLegalizeLmhloFusionToLinalgPass : Pass<"disc-legalize-lmhlo-fusion-to-linalg", "ModuleOp"> {
  let summary = "Pass to convert a lmhlo fusion op to linalg on tensor.";
  let constructor = "createDiscLegalizeLmhloFusionToLinalgPass()";
  let dependentDialects = [
      "disc_ral::disc_linalg_ext::DISCLinalgExtDialect",
  ];
}

def DiscTransformDialectInterpreterPass : Pass<"disc-transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Pass to apply transform dialect operations one by one.";
  let constructor = "createDiscTransformDialectInterpreterPass()";
  let options = [
    Option<"transformFileName_", "transform-file-name", "std::string",
           /*default=*/"\"\"", "Filename of the transform schedule.">,
    Option<"enableExpensiveChecks_", "enable-expensive-checks", "bool",
           /*default=*/"false", "perform expensive checks to better report errors in the transform IR.">,
  ];
}

def DiscTransformDialectEraseSchedulePass : Pass<"disc-transform-dialect-erase-schedule", "ModuleOp"> {
  let summary = "Pass to erase transform dialect schedule from the IR.";
  let constructor = "createDiscTransformDialectEraseSchedulePass()";
}

def DiscRewritePayloadIRForRALPass : Pass<"disc-rewrite-payload-ir-for-ral", "ModuleOp"> {
  let summary = "Pass to rewrite the payload IR transformed by transform IR to be suitable for RAL.";
  let constructor = "createDiscRewritePayloadIRForRALPass()";
  let options = [
    Option<"gpuEnabled_", "gpu-enabled", "bool",
            /*default=*/"false", "whether gpu is available.">,
  ];
}

def DiscMemrefCopyToLinalgPass : Pass<"disc-memref-copy-to-linalg", "func::FuncOp"> {
  let summary = "Pass to convert memref.copy to linalg.";
  let constructor = "createDiscMemrefCopyToLinalgPass()";
  let dependentDialects = [
      "linalg::LinalgDialect",
  ];
}

def DiscConvertForeachThreadOpToParallelOpPass : Pass<"disc-convert-foreach-thread-op-to-parallel-op", "func::FuncOp"> {
  let summary = "Pass to convert memref.copy to linalg.";
  let constructor = "createDiscConvertForeachThreadOpToParallelOpPass()";
}
