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

#include "absl/strings/ascii.h"
#include "llvm/Support/Program.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir {
namespace disc_ral {

namespace {

// The compile trajectory follows the CUDA compilation trajectory,
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cuda-compilation-trajectory
std::vector<std::string> compile_trajectory_template = {
    R"(DISC_CUDA_HOME/nvvm/bin/cicc --c++11 --gnu_version=DISC_GNU_VERSION \
    --allow_managed -arch compute_DISC_SM -m64 --no-version-ident -ftz=1 \
    -prec_div=0 -prec_sqrt=0 -fmad=1 -fast-math --gen_div_approx_ftz \
    --include_file_name "DISC_DIR/DISC_FILE_NAME.fatbin.c" -tused \
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

    for (auto source_code_op : source_code_ops) {
      auto code = source_code_op.getCode().str();
      if (code.empty()) {
        signalPassFailure();
        return;
      }
      cuda_code += code;
      // if (auto attr = source_code_op.getCode()) {
      // cuda_code += attr.getValue();
      // } else {
      // signalPassFailure();
      // return;
      // }
    }

    llvm::errs() << "[ZZ] cuda code to compile:\n" << cuda_code << "\n";

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
  std::string random_number = std::to_string(tensorflow::random::New64());
  std::string tmp_path = "/tmp/";
  bin_path = tmp_path + random_number + ".so";

  // clang-format off
  const char *headers_cpp1 =
#include "tensorflow/compiler/mlir/disc/utils/cutlass_header_preprocess.cpp1.ii.h"
  ;
  const char *headers_cpp4 =
#include "tensorflow/compiler/mlir/disc/utils/cutlass_header_preprocess.cpp4.ii.h"
  ;
  // clang-format on

  std::ofstream cufile_cpp1;
  std::string source_path_cpp1 = tmp_path + random_number + ".cpp1.ii";
  cufile_cpp1.open(source_path_cpp1);
  cufile_cpp1 << headers_cpp1;
  cufile_cpp1 << source;
  cufile_cpp1.close();

  std::ofstream cufile_cpp4;
  std::string source_path_cpp4 = tmp_path + random_number + ".cpp4.ii";
  cufile_cpp4.open(source_path_cpp4);
  cufile_cpp4 << headers_cpp4;
  cufile_cpp4 << source;
  cufile_cpp4.close();

  std::vector<std::string> compile_trajectory = compile_trajectory_template;
  std::string cuda_arch = std::to_string(cc_major_ * 100 + cc_minor_ * 10);
  std::string sm = std::to_string(cc_major_ * 10 + cc_minor_);
  std::string cuda_home;
  if (!executeCommand("dirname $(which nvcc)", &cuda_home) ||
      cuda_home.empty()) {
    return failure();
  }
  absl::StripLeadingAsciiWhitespace(&cuda_home);
  absl::StripTrailingAsciiWhitespace(&cuda_home);
  cuda_home += "/..";
  std::string cuda_version;
  if (!executeCommand("nvcc --version | grep -o 'V[0-9]*\\.[0-9]*\\.[0-9]*'",
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

  for (auto& command : compile_trajectory) {
    stringReplaceInplace(command, "DISC_CUDA_ARCH", cuda_arch, true);
    stringReplaceInplace(command, "DISC_SM", sm, true);
    stringReplaceInplace(command, "DISC_CUDA_MAJOR", cuda_major, true);
    stringReplaceInplace(command, "DISC_CUDA_MINOR", cuda_minor, true);
    stringReplaceInplace(command, "DISC_CUDA_HOME", cuda_home, true);
    stringReplaceInplace(command, "DISC_GNU_VERSION", gnu_version, true);
    stringReplaceInplace(command, "DISC_DIR", tmp_path, true);
    stringReplaceInplace(command, "DISC_FILE_NAME", random_number, true);
  }

  for (auto& command : compile_trajectory) {
    if (!executeCommand(command, nullptr)) {
      return failure();
    }
  }

  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscGPUSourceToLibPass(
    int cc_major, int cc_minor) {
  return std::make_unique<DiscGPUSourceToLibPass>(cc_major, cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir