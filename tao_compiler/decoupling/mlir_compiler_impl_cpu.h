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

#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_CPU_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_CPU_H_

#include "decoupling/mlir_compiler.h"

namespace tensorflow {

namespace tao {

class CompilerMLIR_CPU : public CompilerMLIR {
 public:
  CompilerMLIR_CPU() = default;
  ~CompilerMLIR_CPU() = default;

 private:
  std::string DefaultDevice() override;

  Status FillDeviceInfo(mlir::disc_ral::DISCLoweringOptions& options) override;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_CPU_H_
