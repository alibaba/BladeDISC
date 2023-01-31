//===- cpu_context_impl.h ----------------------===//
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

#ifndef RAL_CONTEXT_BASE_CPU_CPU_CONTEXT_IMPL_H_
#define RAL_CONTEXT_BASE_CPU_CPU_CONTEXT_IMPL_H_

#include "mlir/xla/ral/context/base/base_context.h"
#include "mlir/xla/ral/device/cpu/cpu_driver.h"
#include "mlir/xla/ral/ral_context.h"

namespace tao {
namespace ral {
namespace cpu {
buffer_t cpu_alloc(size_t bytes);
void cpu_dealloc(buffer_t buffer);

struct BaseCpuContextOption {
  std::shared_ptr<Allocator> cpu_allocator;
};

std::unique_ptr<BaseContext> MakeBaseCpuContext(BaseContextOption& opt,
                                                BaseCpuContextOption& cpu_opt);

struct BaseCpuExecutionContext : public tao::ral::BaseExecutionContext {
  BaseCpuExecutionContext(BaseContext* ctx);
  ~BaseCpuExecutionContext();
  // all buffer allocated by the cpu_allocator
  std::unordered_map<const_buffer_t, int> host_ptr_map;

 protected:
  virtual void setOutputDeleter(OutputBufferWrapper& output) override;
};

}  // namespace cpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_BASE_CPU_CPU_CONTEXT_IMPL_H_
