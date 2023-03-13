/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

/**
 *
 ______________________________________
/ You are in a maze of little twisting \
\ passages, all alike.                 /
 --------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
 */
#include <sstream>
#include <unordered_map>

#include "absl/strings/str_split.h"
#include "decoupling/compiler_base.h"
#include "decoupling/tao_compiler_input.pb.h"
#include "decoupling/tao_compiler_trace.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/Timing.h"    // from @llvm-project
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "version.h"

using namespace xla;

namespace tensorflow {

using tensorflow::tao::CompilerBase;

std::unordered_map<std::string, std::string> parse_envs(
    llvm::cl::list<std::string>& envs) {
  std::unordered_map<std::string, std::string> env_pair;
  for (auto& env : envs) {
    std::vector<std::string> kvs = absl::StrSplit(env, '=');
    if (kvs.size() != 2) {
      LOG(FATAL) << "env option value should be ENV=VAL: " << env;
    }
    env_pair[kvs[0]] = kvs[1];
  }
  return env_pair;
}

std::string version_info() {
  std::stringstream ss;
  ss << "DISC Version:"
     << "\n"
     << "TAO_BUILD_VERSION=" << TAO_BUILD_VERSION << "\n"
     << "TAO_BUILD_GIT_HEAD=" << TAO_BUILD_GIT_HEAD << "\n"
     << "TAO_BUILD_TIME=" << TAO_BUILD_TIME;
  return ss.str();
}

Status RealMain(int argc, char** argv) {
  llvm::cl::OptionCategory disc_category("DISC", "Options for DISC.");

  llvm::cl::opt<std::string> input_fn{
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(disc_category)};
  llvm::cl::opt<std::string> output_fn{
      llvm::cl::Positional, llvm::cl::desc("<output file>"),
      llvm::cl::init("/tmp/compilation_result.pbtxt"),
      llvm::cl::cat(disc_category)};

  llvm::cl::opt<std::string> convert_pb_txt{
      "convert-to-pbtxt",
      llvm::cl::desc("convert the protobuf message into txt format."),
      llvm::cl::cat(disc_category)};
  llvm::cl::list<std::string> envs{
      "env",
      llvm::cl::desc("override environmen variables in the input pb file,"
                     "this option can be specified zero or more times."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(disc_category)};

  llvm::cl::opt<bool> version("v", llvm::cl::desc("show DIS compiler version."),
                              llvm::cl::cat(disc_category));
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream& os) { os << version_info(); });

  llvm::cl::HideUnrelatedOptions(disc_category);
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "Welcome DISC!\n");

  if (version) {
    std::cout << version_info() << std::endl;
    return tsl::OkStatus();
  }

  tensorflow::tao::TaoCompilerInput input;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), input_fn, &input));

  if (!convert_pb_txt.empty()) {
    VLOG(0) << "the output protobuf message file with text format: "
            << output_fn;
    return WriteTextProto(Env::Default(), output_fn, input);
  }

  if (!input.env().empty()) {
    VLOG(1) << "Setting up environment variable compiler:";
    for (auto& kv : input.env()) {
      setenv(kv.first.c_str(), kv.second.c_str(), 1);
      VLOG(1) << "    " << kv.first << "=" << kv.second;
      std::string event_value = strings::StrCat(kv.first, "=", kv.second);
      tao::TaoCompilerTrace::Instance()->OnEvent("SetEnvVar",
                                                 std::move(event_value));
    }
  }

  if (!envs.empty()) {
    auto cmd_envs = parse_envs(envs);
    VLOG(1) << "Setting up environment variable from cmdline:";
    for (auto& kv : cmd_envs) {
      VLOG(1) << "    " << kv.first << "=" << kv.second;
      setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
  }

  DeviceType device_type(input.options().device_type());
  auto status_or = CompilerBase::GetCompilerForDevice(device_type);
  if (!status_or.ok()) return status_or.status();
  auto* compiler_wrapper = status_or.value();
  TF_RETURN_IF_ERROR(compiler_wrapper->Compile(input, output_fn));
  tao::TaoCompilerTrace::Instance()->Shutdown();
  return tsl::OkStatus();
}

}  // namespace tensorflow

int main(int argc, char** argv) {
  std::vector<tsl::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  auto status = tensorflow::RealMain(argc, argv);
  if (status.ok()) {
    VLOG(0) << "Success!";
    return 0;
  } else {
    std::string err_msg = status.error_message();
    tensorflow::error::Code code = status.code();
    VLOG(0) << "Failed! " << err_msg << " code " << code;
    if (code == tensorflow::error::RESOURCE_EXHAUSTED) {
      return 2;
    } else if (code == tensorflow::error::DEADLINE_EXCEEDED) {
      return 3;
    }
    return 1;
  }
}
