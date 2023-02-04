// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_GPU_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_GPU_H_

#include <memory>

#include "decoupling/mlir_compiler.h"

namespace tensorflow {

namespace tao {

class CompilerMLIR_GPU : public CompilerMLIR {
 public:
  CompilerMLIR_GPU();
  ~CompilerMLIR_GPU();

 private:
  Status Init(const TaoCompilerInput& input,
              const string& output_file) override;

  std::string DefaultDevice() override;

  Status FillDeviceInfo(mlir::disc_ral::DISCLoweringOptions& options) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_DHLO_COMPILER_IMPL_GPU_H_
