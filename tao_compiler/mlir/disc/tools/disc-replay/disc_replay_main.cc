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

#include "llvm/Support/CommandLine.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"
#include "tensorflow/core/platform/errors.h"

tensorflow::Status RealMain(int argc, char** argv) {
  llvm::cl::OptionCategory replay_cat("DISC Replay Toolkit",
                                      "Options for DISC Replay.");

  llvm::cl::opt<std::string> program_fname{
      "p", llvm::cl::Required,
      llvm::cl::desc("The tao_compiler_input protobuf message."),
      llvm::cl::value_desc("program"), llvm::cl::cat(replay_cat)};
  llvm::cl::opt<std::string> data_fname{
      "d", llvm::cl::Required, llvm::cl::desc("The compressed input tensors."),
      llvm::cl::value_desc("data"), llvm::cl::cat(replay_cat)};
  llvm::cl::HideUnrelatedOptions(replay_cat);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Welcome BladeDISC!\n");

  if (program_fname.empty() || data_fname.empty()) {
    return tensorflow::Status::OK();
  }
  auto record = replay::CreateReplayRecord(program_fname, data_fname);
  if (record == nullptr) {
    return tensorflow::errors::Internal("load replay record failed!");
  }

  replay::DiscInterpreter disc;
  replay::CompiledResult result;
  TF_RETURN_IF_ERROR(disc.Compile(record->Program(), result));
  TF_RETURN_IF_ERROR(disc.Run(result, record->Tensors(), record->Placements()));
  return tensorflow::Status::OK();
}

int main(int argc, char** argv) {
  auto s = RealMain(argc, argv);
  if (!s.ok()) {
    VLOG(0) << "replay failed " << s.error_message();
    return 1;
  }
  return 0;
}