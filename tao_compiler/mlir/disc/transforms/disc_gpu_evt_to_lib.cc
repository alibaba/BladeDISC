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

#include <cstdio>
#include <fstream>
#include <unordered_map>

#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "llvm/Support/Program.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/platform/cuda_libdevice_path.h"
#include "tensorflow/tsl/platform/env.h"

namespace mlir {
namespace disc_ral {

namespace {

std::string compile_cmd = "python3 DISC_DIR/DISC_FILE_NAME.py";

class DiscGPUEvtSourceToLibPass
    : public DiscGPUEvtSourceToLibPassBase<DiscGPUEvtSourceToLibPass> {
 public:
  explicit DiscGPUEvtSourceToLibPass(int cc_major, int cc_minor)
      : DiscGPUEvtSourceToLibPassBase<
            DiscGPUEvtSourceToLibPass>::DiscGPUEvtSourceToLibPassBase() {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Find ops containing CUDA code and concatenate CUDA code.
    std::string cuda_code;
    SmallVector<lmhlo_disc::EvtSourceOp> source_code_ops;
    m.walk([&](lmhlo_disc::EvtSourceOp source_code_op) {
      source_code_ops.push_back(source_code_op);
    });

    if (source_code_ops.empty()) {
      return;
    }

    for (auto source_code_op : source_code_ops) {
      auto code = source_code_op.getCode().str();
      if (code.empty()) {
        continue;
      }
      std::string bin_path;
      if (failed(compilePreprocessedCUDAEvtToLib(code, bin_path))) {
        signalPassFailure();
        return;
      }
      OpBuilder builder(source_code_op);
      source_code_op->setAttr(kDynLibPathAttr, builder.getStringAttr(bin_path));
    }
  }

 private:
  int cc_major_;
  int cc_minor_;

 private:
  // Return true on success, false otherwise.
  bool executeCommand(const std::string& cmd, std::string* output);
  LogicalResult compilePreprocessedCUDAEvtToLib(const std::string& source,
                                                std::string& bin_path);
  std::string findCUDAHome();
};

bool DiscGPUEvtSourceToLibPass::executeCommand(const std::string& cmd,
                                         std::string* output) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return false;
  }

  if (output) {
    char tmp_buffer[128];
    while (fgets(tmp_buffer, sizeof(tmp_buffer), pipe)) {
      output->append(tmp_buffer);
    }
  }
  return pclose(pipe) == 0;
}

LogicalResult DiscGPUEvtSourceToLibPass::compilePreprocessedCUDAEvtToLib(
    const std::string& source, std::string& bin_path) {
#if defined(GOOGLE_CUDA)
#ifndef SKIP_COMPUTE_INTENSIVE_FUSION

  std::string random_number = std::to_string(tensorflow::random::New64());
  std::string tmp_path = "/tmp/";
  bin_path = tmp_path + random_number + ".so";

  auto synthesizeCodeAndWriteFile = [&](std::string path, std::string body) {
    std::ofstream file;
    file.open(path);
    file << body;
    file.close();
  };

  synthesizeCodeAndWriteFile(tmp_path + random_number + ".py", source);

  std::unordered_map<std::string, std::string> argToReplace = {
      {"DISC_DIR", tmp_path}, {"DISC_FILE_NAME", random_number}};

  compile_cmd = absl::StrReplaceAll(compile_cmd, argToReplace);

  if (executeCommand(compile_cmd, nullptr)) {
    return success();
  }

  return success();
#else

  return failure();

#endif  // SKIP_COMPUTE_INTENSIVE_FUSION

#else

  return failure();

#endif  // GOOGLE_CUDA
}

std::string DiscGPUEvtSourceToLibPass::findCUDAHome() {
  for (const std::string& cuda_root : tsl::CandidateCudaRoots()) {
    std::string path = tensorflow::io::JoinPath(cuda_root, "bin", "nvcc");
    VLOG(2) << "Looking for nvcc at " << path;
    if (tsl::Env::Default()->FileExists(path).ok()) {
      VLOG(2) << "Found nvcc in cuda home " << cuda_root;
      return cuda_root;
    }
  }
  return "";
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscGPUEvtSourceToLibPass(
    int cc_major, int cc_minor) {
  return std::make_unique<DiscGPUEvtSourceToLibPass>(cc_major, cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir
