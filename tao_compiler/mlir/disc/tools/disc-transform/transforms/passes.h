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

#ifndef DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSES_H_
#define DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

namespace mlir {

class FunctionPass;
class ModuleOp;
class Operation;
class Pass;

namespace func {
class FuncOp;
}

template <typename T>
class OperationPass;

namespace disc_ral {

namespace disc_linalg_ext {
class DISCLinalgExtDialect;
}

// Converts a lmhlo fusion op in side a function to its linalg on tensor
// equivalent.
std::unique_ptr<OperationPass<ModuleOp>>
createDiscLegalizeLmhloFusionToLinalgPass();

// Applys transform dialect ops for codegen.
std::unique_ptr<OperationPass<ModuleOp>>
createDiscTransformDialectInterpreterPass(const std::string& fileName = "",
                                          bool enableExpensiveChecks = false);

// Erases transform dialect schedule from the IR
std::unique_ptr<OperationPass<ModuleOp>>
createDiscTransformDialectEraseSchedulePass();

// Converts the transformed payload IR to be suitable for RAL.
std::unique_ptr<OperationPass<ModuleOp>> createDiscRewritePayloadIRForRALPass(
    bool gpuEnabled = false);

// Converts a memref.copy op to its linalg equivalent
std::unique_ptr<OperationPass<func::FuncOp>> createDiscMemrefCopyToLinalgPass();

// Converts scf.foreach_thread to scf.parallel
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertForeachThreadOpToParallelOpPass();

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSES_H_
