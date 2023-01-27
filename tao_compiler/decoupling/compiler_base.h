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

#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_

#include "decoupling/tao_compilation_result.pb.h"
#include "decoupling/tao_compiler_input.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

class CompilerBase {
 public:
  virtual Status Compile(const tensorflow::tao::TaoCompilerInput& input,
                         const string& output_file) = 0;

  using CompilerFactory = std::function<std::unique_ptr<CompilerBase>()>;

  static void RegisterCompilerFactory(DeviceType dt, CompilerFactory factory);

  static StatusOr<CompilerBase*> GetCompilerForDevice(DeviceType dt);
};

}  //  namespace tao
}  //  namespace tensorflow
#endif  //  TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_