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

#ifndef DISC_DISC_COMPILER_H_
#define DISC_DISC_COMPILER_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"

namespace mlir {
namespace disc_ral {

enum CodeGenMode {
  // Do CPU CodeGen for all mhlo ops.
  kCpuCentric,
  // Do GPU CodeGen for all mhlo ops if possible
  kGpuCentric
};

struct GpuDeviceInfo {
  int cc_major = -1;
  int cc_minor = -1;
  int sm_count = -1;
  int max_threads_per_sm = -1;
  int device_ordinal = 0;
};

struct GpuLoweringOptions {
  bool multi_cc_support = false;
  bool multi_cc_support_dbg_ptx_only = false;
};

struct CpuLoweringOptions {
  explicit CpuLoweringOptions(bool init_from_env_vars = true);

  // init options from env vars
  void initFromEnvVars();

  // Level 0: no fast math
  // Level 1: apply approximation for some expensive math ops (e.g. exp, sin)
  // Level 2: Level 1 + AllowReassoc
  // Level 3: Level 2 + NoNaNs + NoSignedZeros
  // Level 4: Level 3 + fully llvm fast math
  int64_t fast_math_level = 1;

  // -1 means using the default strategy
  // supported values: 128/256/512
  int64_t vector_width = -1;

  // If true, disable llvm to do loop unroll
  bool disable_loop_unroll = false;

  // If true, assume all buffers are not overlapped.
  bool assume_no_buffer_alias = true;

  // If true, codegen for multi threading execution environment
  bool target_multi_threading = true;

  std::string llvm_target_triple = "";
  std::string llvm_target_cpu = "";
  std::string llvm_target_cpu_features = "";
};

struct DISCLoweringOptions {
  DISCLoweringOptions(const std::string& output_file_name,
                      CodeGenMode mode = kGpuCentric)
      : output_file_name(output_file_name),
        metadata_file_path(output_file_name + ".pbtxt"),
        mode(mode) {}
  std::string output_file_name;
  std::string metadata_file_path;
  CodeGenMode mode;
  GpuDeviceInfo gpu_info;
  GpuLoweringOptions gpu_options;
  CpuLoweringOptions cpu_options;
};

LogicalResult LowerHLOToLLVM(ModuleOp m, const DISCLoweringOptions& options);

LogicalResult LowerLLVMToBinary(ModuleOp m, const DISCLoweringOptions& options,
                                std::string& out);

LogicalResult BinaryStrToSharedLibrary(const DISCLoweringOptions& options,
                                       const std::string& binary);

LogicalResult LowerHLOToSharedLibrary(ModuleOp m,
                                      const DISCLoweringOptions& options);

}  // namespace disc_ral
}  // namespace mlir

namespace tensorflow {

Status ConvertTF2MlirHlo(mlir::ModuleOp module_op);

}  // namespace tensorflow

#endif  // DISC_DISC_COMPILER_H_
