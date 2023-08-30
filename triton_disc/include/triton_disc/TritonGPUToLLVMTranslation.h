// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRITON_DISC_TARGET_LLVMIRTRANSLATION_H
#define TRITON_DISC_TARGET_LLVMIRTRANSLATION_H
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
class LLVMContext;
}  // namespace llvm

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace mlir {
namespace triton_disc {

// Translate TritonGPU dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module> translateTritonGPUToLLVMIR(
    llvm::LLVMContext* llvmContext, mlir::ModuleOp module,
    int computeCapability);

}  // namespace triton_disc
}  // namespace mlir

#endif  // TRITON_TARGET_LLVMIRTRANSLATION_H
