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

#ifndef TRITON_GPU_TO_MLIR_GPU_H
#define TRITON_GPU_TO_MLIR_GPU_H
#include <memory>

namespace mlir {
class ModuleOp;
namespace func {
class FuncOp;
}
template <typename T>
class OperationPass;

namespace triton_disc {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createTritonGPUToMLIRGPUPass();
}  //   namespace triton_disc

}  //   namespace mlir
#endif