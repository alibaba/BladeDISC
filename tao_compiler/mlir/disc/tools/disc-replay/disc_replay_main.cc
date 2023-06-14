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

#if GOOGLE_CUDA
#include <cuda_profiler_api.h>
#endif

#include "llvm/Support/CommandLine.h"
#include "mlir/disc/tools/disc-replay/disc_interpreter.h"
#include "tensorflow/core/platform/errors.h"

tensorflow::Status RealMain(int argc, char** argv) {
  llvm::cl::OptionCategory replay_cat("DISC Replay Toolkit",
                                      "Options for DISC Replay.");

  llvm::cl::opt<std::string> program_fname{
      "p", llvm::cl::Required,
      llvm::cl::desc("The tao_compiler_input protobuf message."),
      llvm::cl::value_desc("string"), llvm::cl::cat(replay_cat)};
  llvm::cl::opt<std::string> data_fname{
      "d", llvm::cl::Required, llvm::cl::desc("The compresses input tensors."),
      llvm::cl::value_desc("string"), llvm::cl::cat(replay_cat)};
  llvm::cl::opt<int> warmup_iters{"w", llvm::cl::desc("warmup iterations"),
                                  llvm::cl::value_desc("int"),
                                  llvm::cl::init(5), llvm::cl::cat(replay_cat)};
  llvm::cl::opt<bool> enable_nvprof{
      "enable-nvprof", llvm::cl::desc("enable nvprof or not, default is false"),
      llvm::cl::value_desc("bool"), llvm::cl::init(false),
      llvm::cl::cat(replay_cat)};

  llvm::cl::HideUnrelatedOptions(replay_cat);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Welcome BladeDISC!\n");

  auto record = replay::CreateReplayRecord(program_fname, data_fname);
  if (record == nullptr) {
    return tensorflow::errors::Internal("load replay record failed!");
  }

  replay::DiscInterpreter disc;
  replay::CompiledResult result;
  TF_RETURN_IF_ERROR(disc.Compile(record->Program(), result));

  for (int i = 0; i < warmup_iters; ++i) {
    TF_RETURN_IF_ERROR(
        disc.Run(result, record->Tensors(), record->Placements()));
  }
  VLOG(0) << "Finish warmup with " << warmup_iters << " iterations";

#if GOOGLE_CUDA
  if (enable_nvprof) cudaProfilerStart();
#endif
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  TF_RETURN_IF_ERROR(disc.Run(result, record->Tensors(), record->Placements()));
  VLOG(0) << "Replay toolkit execution uses: "
          << std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - begin)
                     .count() /
                 1000.0
          << " ms";
#if GOOGLE_CUDA
  if (enable_nvprof) cudaProfilerStop();
#endif
  return tsl::OkStatus();
}

int main(int argc, char** argv) {
  auto s = RealMain(argc, argv);
  if (!s.ok()) {
    VLOG(0) << "replay failed " << s.error_message();
    return 1;
  }
  return 0;
}
