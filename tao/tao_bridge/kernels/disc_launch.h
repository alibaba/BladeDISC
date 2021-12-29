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

#ifndef TAO_TAO_BRIDGE_KERNELS_DISC_LAUNCH_H_
#define TAO_TAO_BRIDGE_KERNELS_DISC_LAUNCH_H_

#include "tao_bridge/common.h"
#include "tao_bridge/kernels/launch_base.h"
#include "tao_bridge/kernels/platform_info.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace tao {

struct TaoProfileStat;

class DiscLaunchOp : public LaunchBase {
 public:
  explicit DiscLaunchOp(OpKernelConstruction* ctx);
  ~DiscLaunchOp() override;

  void ComputeAsync(OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

 private:
  Status CompileAndRunMlir(OpKernelContext*, DoneHelper*, bool*);

  Status CompileToLocalExecutable(OpKernelContext* ctx,
                                  const NameAttrList& function, bool is_mlir,
                                  TaoCompileFuncCallInfo* call_info,
                                  std::map<int, OptionalTensor>* variables,
                                  tao::Executable** executable,
                                  TaoProfileStat** stat);

  NameAttrList mlir_function_;
  FunctionLibraryRuntime::Handle mlir_func_handle_;

  // Currently non-empty for GPU only
  bool enable_fallback_;
  bool verbose_compilation_log_;

  // async or sync compilation
  CompilationMode mode_;
  // stash mlir compile status
  Status mlir_compile_status_;
  std::shared_ptr<TaoLaunchTick> tick_;

  std::atomic<int> err_msg_print_counter_{0};
  TF_DISALLOW_COPY_AND_ASSIGN(DiscLaunchOp);
};

#define REGISTER_DISC_LAUNCH_KERNEL(DEVICE)              \
  REGISTER_KERNEL_BUILDER(Name("DiscLaunch")             \
                              .Device(DEVICE)            \
                              .HostMemory("constants")   \
                              .HostMemory("fixedshapes") \
                              .HostMemory("hostargs")    \
                              .HostMemory("hostresults") \
                              .HostMemory("resources"),  \
                          DiscLaunchOp)

}  //  namespace tao
}  //  namespace tensorflow
#endif  //  TAO_TAO_BRIDGE_KERNELS_DISC_LAUNCH_H_