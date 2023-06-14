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

#ifndef DISC_REPLAY_DISC_INTERPRETER_H_
#define DISC_REPLAY_DISC_INTERPRETER_H_

#include "decoupling/mlir_compiler.h"
#include "mlir/disc/tools/disc-replay/record.h"
#include "mlir/ral/ral_api.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "mlir/ral/context/base/cuda/cuda_context_impl.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#else
#include "mlir/ral/context/base/cpu/cpu_context_impl.h"
#endif

namespace tensorflow {
class Status;
}  // namespace tensorflow

namespace replay {
// function type for compilation executable file
using func_t = void (*)(void**);

struct CompiledResult {
  // compiled result is a dynamic library file with suffix ".so"
  std::string output_fname;
  // meta file is a protobuf file with suffix ".pbtxt"
  std::string meta_fname;
  // entry function for the executable program
  func_t entry_func;
};

// DiscInterpreter implements an interatpre for BladeDISC,
// which compiles input program (tao_compiler_input.proto) into an executable
// program, and run with input data (tensors).
//
// Example:
//
//    DiscInterpreter disc;
//    CompiledResult result;
//    auto record = CreateRecord(input_prog, data_tarball);
//    disc.Compile(record->Program(), result);
//    disc.Run(result, record->Tensors(), record->Placements());
class DiscInterpreter {
 public:
  DiscInterpreter();
  ~DiscInterpreter(){};
  // Compile takes DISC program and outputs the executable program which wrapped
  // into CompiledResult
  tensorflow::Status Compile(tensorflow::tao::TaoCompilerInput& input,
                             CompiledResult& result);
  // Run the executable program with input data
  tensorflow::Status Run(const CompiledResult& result,
                         const std::vector<tensorflow::Tensor>& tensors,
                         const std::vector<std::string>& placements);

 private:
  void InitExecCUDAContext(const std::string& executable_fname);
  tensorflow::Status GetEntryFunc(const std::string& exectuable_fname,
                                  func_t& entry_func);

  void* ral_func_ptr_;
  std::unique_ptr<tao::ral::BaseContext> context_;
};

}  //  namespace replay

#endif  // DISC_REPLAY_DISC_INTERPRETER_H_
