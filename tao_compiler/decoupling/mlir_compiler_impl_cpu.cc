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

#include "decoupling/mlir_compiler_impl_cpu.h"

namespace tensorflow {
namespace tao {

std::string CompilerMLIR_CPU::DefaultDevice() { return "cpu"; }

Status CompilerMLIR_CPU::FillDeviceInfo(
    mlir::disc_ral::DISCLoweringOptions& options) {
  // TODO(kevin.zwy): add an cpu_info field to store #cores configuration.
  options.mode = mlir::disc_ral::CodeGenMode::kCpuCentric;

  return tsl::OkStatus();
}

}  // namespace tao
}  // namespace tensorflow

static bool InitModule() {
  tensorflow::tao::CompilerBase::RegisterCompilerFactory(
      "MLIR_CPU", []() -> std::unique_ptr<tensorflow::tao::CompilerBase> {
        return absl::make_unique<tensorflow::tao::CompilerMLIR_CPU>();
      });
  return true;
}
static bool module_initialized = InitModule();
