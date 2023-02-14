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

// The compile trajectory follows the CUDA compilation trajectory,
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cuda-compilation-trajectory
std::vector<std::string> compile_trajectory_template = {
    R"(DISC_CUDA_HOME/nvvm/bin/cicc --c++11 --gnu_version=DISC_GNU_VERSION \
    --allow_managed -arch compute_DISC_SM -m64 -ftz=1 \
    -prec_div=0 -prec_sqrt=0 -fmad=1 -fast-math --gen_div_approx_ftz \
    --include_file_name "DISC_DIR/DISC_FILE_NAME.fatbin.c" -tused \
    -nvvmir-library "/usr/local/cuda/bin/../nvvm/libdevice/libdevice.10.bc" \
    --gen_module_id_file \
    --module_id_file_name "DISC_DIR/DISC_FILE_NAME.module_id" \
    --gen_c_file_name "DISC_DIR/DISC_FILE_NAME.cudafe1.c" \
    --stub_file_name "DISC_DIR/DISC_FILE_NAME.cudafe1.stub.c" \
    --gen_device_file_name "DISC_DIR/DISC_FILE_NAME.cudafe1.gpu" \
    "DISC_DIR/DISC_FILE_NAME.cpp1.ii" \
    -o "DISC_DIR/DISC_FILE_NAME.ptx" )",

    R"(ptxas -arch=sm_DISC_SM -m64 "DISC_DIR/DISC_FILE_NAME.ptx" \
    -o "DISC_DIR/DISC_FILE_NAME.cubin" )",

    R"(fatbinary --create="DISC_DIR/DISC_FILE_NAME.fatbin" -64 \
    --cicc-cmdline="-ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1 " \
    "--image3=kind=elf,sm=DISC_SM,file=DISC_DIR/DISC_FILE_NAME.cubin" \
    --embedded-fatbin="DISC_DIR/DISC_FILE_NAME.fatbin.c" )",

    R"(cudafe++ --c++11 --gnu_version=DISC_GNU_VERSION --allow_managed --m64 \
    --parse_templates \
    --gen_c_file_name "DISC_DIR/DISC_FILE_NAME.cudafe1.cpp" \
    --stub_file_name "DISC_DIR/DISC_FILE_NAME.cudafe1.stub.c" \
    --module_id_file_name "DISC_DIR/DISC_FILE_NAME.module_id" \
    "DISC_DIR/DISC_FILE_NAME.cpp4.ii" )",

    R"(gcc -std=c++11 -D__CUDA_ARCH__=DISC_CUDA_ARCH -c -x c++ \
    -DCUDA_DOUBLE_MATH_FUNCTIONS -fPIC -O3 \
    "-IDISC_CUDA_HOME/targets/x86_64-linux/include" -m64 \
    "DISC_DIR/DISC_FILE_NAME.cudafe1.cpp" \
    -o "DISC_DIR/DISC_FILE_NAME.o" )",

    R"(nvlink --arch=sm_DISC_SM \
    --register-link-binaries="DISC_DIR/DISC_FILE_NAME_dlink.reg.c" -m64 \
    -L"DISC_CUDA_HOME/lib64" "-LDISC_CUDA_HOME/targets/x86_64-linux/lib/stubs" \
    "-LDISC_CUDA_HOME/targets/x86_64-linux/lib" -cpu-arch=X86_64 \
    "DISC_DIR/DISC_FILE_NAME.o" -lcudadevrt \
    -o "DISC_DIR/DISC_FILE_NAME_dlink.cubin" )",

    R"(fatbinary --create="DISC_DIR/DISC_FILE_NAME_dlink.fatbin" -64 \
    --cicc-cmdline="-ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1 " -link \
    "--image3=kind=elf,sm=DISC_SM,file=DISC_DIR/DISC_FILE_NAME_dlink.cubin" \
    --embedded-fatbin="DISC_DIR/DISC_FILE_NAME_dlink.fatbin.c" )",

    R"(gcc -std=c++11 -c -x c++ -fPIC \
    -DFATBINFILE="\"DISC_DIR/DISC_FILE_NAME_dlink.fatbin.c\"" \
    -DREGISTERLINKBINARYFILE="\"DISC_DIR/DISC_FILE_NAME_dlink.reg.c\"" \
    -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= \
    -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ -fPIC -O3 \
    "-IDISC_CUDA_HOME/targets/x86_64-linux/include" \
    -D__CUDACC_VER_MAJOR__=DISC_CUDA_MAJOR \
    -D__CUDACC_VER_MINOR__=DISC_CUDA_MINOR \
    -D__CUDA_API_VER_MAJOR__=DISC_CUDA_MAJOR \
    -D__CUDA_API_VER_MINOR__=DISC_CUDA_MINOR -m64 \
    "DISC_CUDA_HOME/bin/crt/link.stub" \
    -o "DISC_DIR/DISC_FILE_NAME_dlink.o" )",

    R"(g++ -fPIC -O3 -m64 -std=c++11 -Wl,--start-group \
    "DISC_DIR/DISC_FILE_NAME_dlink.o" "DISC_DIR/DISC_FILE_NAME.o" \
    -shared -L"DISC_CUDA_HOME/lib64" \
    "-LDISC_CUDA_HOME/targets/x86_64-linux/lib/stubs" \
    "-LDISC_CUDA_HOME/targets/x86_64-linux/lib" -lcudadevrt -lcudart_static \
    -lrt -lpthread -ldl -Wl,--end-group -o "DISC_DIR/DISC_FILE_NAME.so" )"};

class DiscGPUSourceToLibPass
    : public DiscGPUSourceToLibPassBase<DiscGPUSourceToLibPass> {
 public:
  explicit DiscGPUSourceToLibPass(int cc_major, int cc_minor)
      : DiscGPUSourceToLibPassBase<
            DiscGPUSourceToLibPass>::DiscGPUSourceToLibPassBase() {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Find ops containing CUDA code and concatenate CUDA code.
    std::string cuda_code;
    SmallVector<lmhlo_disc::SourceCodeOp> source_code_ops;
    m.walk([&](lmhlo_disc::SourceCodeOp source_code_op) {
      source_code_ops.push_back(source_code_op);
    });

    if (source_code_ops.empty()) {
      return;
    }

    for (auto source_code_op : source_code_ops) {
      auto code = source_code_op.getCode().str();
      if (code.empty()) {
        signalPassFailure();
        return;
      }
      cuda_code += code;
    }

    // Compile CUDA code.
    std::string bin_path;
    if (failed(compilePreprocessedCUDASourceToLib(cuda_code, bin_path))) {
      signalPassFailure();
      return;
    }

    for (auto source_code_op : source_code_ops) {
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
  LogicalResult compilePreprocessedCUDASourceToLib(const std::string& source,
                                                   std::string& bin_path);
  std::string findCUDAHome();
};

bool DiscGPUSourceToLibPass::executeCommand(const std::string& cmd,
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

LogicalResult DiscGPUSourceToLibPass::compilePreprocessedCUDASourceToLib(
    const std::string& source, std::string& bin_path) {
#if defined(GOOGLE_CUDA)
#ifndef SKIP_COMPUTE_INTENSIVE_FUSION

  std::string random_number = std::to_string(tensorflow::random::New64());
  std::string tmp_path = "/tmp/";
  bin_path = tmp_path + random_number + ".so";

  char* headers_cpp1;
  if (cc_major_ == 7 && cc_minor_ == 0) {
    headers_cpp1 =
#include "mlir/disc/utils/cutlass_header_preprocess.700.cpp1.ii.h"
        ;
  } else if (cc_major_ == 7 && cc_minor_ == 5) {
    headers_cpp1 =
#include "mlir/disc/utils/cutlass_header_preprocess.750.cpp1.ii.h"
        ;
  } else if (cc_major_ == 8 && cc_minor_ == 0) {
    headers_cpp1 =
#include "mlir/disc/utils/cutlass_header_preprocess.800.cpp1.ii.h"
        ;
  } else if (cc_major_ == 8 && cc_minor_ == 6) {
    headers_cpp1 =
#include "mlir/disc/utils/cutlass_header_preprocess.860.cpp1.ii.h"
        ;
  } else {
    return failure();
  }

  const char* headers_cpp4 =
#include "mlir/disc/utils/cutlass_header_preprocess.cpp4.ii.h"
      ;

  auto synthesizeCodeAndWriteFile = [&](std::string path, std::string header,
                                        std::string body) {
    std::ofstream file;
    file.open(path);
    file << header;
    file << body;
    file.close();
  };

  synthesizeCodeAndWriteFile(tmp_path + random_number + ".cpp1.ii",
                             headers_cpp1, source);
  synthesizeCodeAndWriteFile(tmp_path + random_number + ".cpp4.ii",
                             headers_cpp4, source);

  std::vector<std::string> compile_trajectory = compile_trajectory_template;
  std::string cuda_arch = std::to_string(cc_major_ * 100 + cc_minor_ * 10);
  std::string sm = std::to_string(cc_major_ * 10 + cc_minor_);
  std::string cuda_home = findCUDAHome();
  if (cuda_home.empty()) {
    return failure();
  }
  std::string cuda_version;
  std::string nvcc_path = tensorflow::io::JoinPath(cuda_home, "bin", "nvcc");
  if (!executeCommand(nvcc_path + " --version | grep -o "
                                  "'V[0-9]*\\.[0-9]*\\.[0-9]*'",
                      &cuda_version) ||
      cuda_version.empty()) {
    return failure();
  }
  absl::StripLeadingAsciiWhitespace(&cuda_version);
  absl::StripTrailingAsciiWhitespace(&cuda_version);
  cuda_version = cuda_version.substr(1);
  auto cuda_version_num_str = tensorflow::str_util::Split(cuda_version, '.');
  std::string cuda_major = cuda_version_num_str[0];
  std::string cuda_minor = cuda_version_num_str[1];
  std::string gnu_version;
  std::string gnu_ver_cmd =
      R"(gcc -dumpfullversion -dumpversion |
          sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' \
              -e 's/^[0-9]\{3,4\}$/&00/')";
  if (!executeCommand(gnu_ver_cmd, &gnu_version) || gnu_version.empty()) {
    return failure();
  }
  absl::StripLeadingAsciiWhitespace(&gnu_version);
  absl::StripTrailingAsciiWhitespace(&gnu_version);

  std::unordered_map<std::string, std::string> argToReplace = {
      {"DISC_CUDA_ARCH", cuda_arch},   {"DISC_SM", sm},
      {"DISC_CUDA_MAJOR", cuda_major}, {"DISC_CUDA_MINOR", cuda_minor},
      {"DISC_CUDA_HOME", cuda_home},   {"DISC_GNU_VERSION", gnu_version},
      {"DISC_DIR", tmp_path},          {"DISC_FILE_NAME", random_number}};

  for (auto& command : compile_trajectory) {
    command = absl::StrReplaceAll(command, argToReplace);
  }

  for (auto& command : compile_trajectory) {
    if (!executeCommand(command, nullptr)) {
      return failure();
    }
  }

  return success();
#else

  return failure();

#endif  // SKIP_COMPUTE_INTENSIVE_FUSION

#else

  return failure();

#endif  // GOOGLE_CUDA
}

std::string DiscGPUSourceToLibPass::findCUDAHome() {
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

std::unique_ptr<OperationPass<ModuleOp>> createDiscGPUSourceToLibPass(
    int cc_major, int cc_minor) {
  return std::make_unique<DiscGPUSourceToLibPass>(cc_major, cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir
