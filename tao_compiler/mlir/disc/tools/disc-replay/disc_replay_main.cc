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
#include "tensorflow/core/platform/error.h"

tensorflow::Status RealMain(int argc, char** argv) {
  llvm::cl::opt<std::string> program_fname{llvm::cl::desc("<program file>")};
  llvm::cl::opt<std::string> data_fname{llvm::cl::desc("<data tar file")};

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Thanks for using DISC Replay Toolkit!\n");

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