//===- tf_context_impl.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#ifndef RAL_CONTEXT_TENSORFLOW_TF_CONTEXT_IMPL_H_
#define RAL_CONTEXT_TENSORFLOW_TF_CONTEXT_IMPL_H_

#include <mutex>

#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_context.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

template <typename T>
inline bool ral_to_bool(const T& condition) {
  return condition;
}

template <>
inline bool ral_to_bool(const Status& status) {
  return status.ok();
}

struct RalTfContextOptions {
  std::string metadata_file_path;
};

class RalTfContext : public ::tao::ral::Context {
 public:
  RalTfContext(const RalTfContextOptions& options = RalTfContextOptions());
  ~RalTfContext();
};

class RalTfExecutionContext : public ::tao::ral::ExecutionContext {
 public:
  RalTfExecutionContext(RalTfContext* ctx);
  ~RalTfExecutionContext();

  OpKernelContext* getOpContext();
  void setOpContext(OpKernelContext* ctx);

  struct Impl;
  Impl* getImpl() { return impl_.get(); };

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace tensorflow

#endif  // RAL_CONTEXT_TENSORFLOW_TF_CONTEXT_IMPL_H_
