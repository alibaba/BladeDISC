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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_H_

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 128
#endif

#include <chrono>

#ifdef DISC_BUILD_FROM_TF_BRIDGE
#include "tensorflow/compiler/mlir/xla/compile_metadata.pb.h"
#else
#include "tensorflow/compiler/mlir/xla/ral/compile_metadata.pb.h"
#endif
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_context.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"

namespace mlir {
class MetadataProto;
}

namespace tao {
namespace ral {

extern const char* kRalGlobalConstantState;
extern const std::map<std::pair<int, int>, size_t> c_CC_INDEX_MAP;

struct ProcessLevelConstStore;
struct ConstStoreRegistrar {
  static ConstStoreRegistrar& Instance();

  bool unregisterConstStore(ProcessLevelConstStore* store);
  ProcessLevelConstStore* getConstStore(std::string pb_file_path);

  std::mutex mu;
  std::unordered_map<std::string, ProcessLevelConstStore*> pbFile2Instance;
  std::unordered_map<ProcessLevelConstStore*, int> referenceCounter;
};

struct RalGlobalConstantState : public tao::ral::Context::Resource {
  std::recursive_mutex mu;
  mlir::MetadataProto metadata_proto;
  // If not null, use the process level const store instead of this context
  // level store
  ProcessLevelConstStore* process_level_store = nullptr;
  // map <unique_name, <device/host_ptr, shape>>
  // in theory, there might be two constants with the same data value but
  // different shapes however for simplicity we just use the whole unique_name
  // as the key. This can be further optimized in case neccessary.
  std::map<std::string, std::pair<buffer_t, buffer_shape_t>> device_constants;
  std::map<std::string, std::pair<buffer_t, buffer_shape_t>> host_constants;

  void onContextFinish(Context* ctx) override;
};

struct ProcessLevelConstStore {
  RalGlobalConstantState state;
  std::string pb_file_path;
};

// Enables process level const store if true.
bool discEnableGlobalConstantStore();

int parseMetadataPb(const std::string& pb_file_path,
                    mlir::MetadataProto* proto);

template <typename T, int N>
MemRefType<T, N> ral_base_cuda_const_cuda(ExecutionContext* ctx,
                                          void* stream_handle,
                                          const char* unique_name);

template <typename T>
MemRefType<T, 0> ral_base_cuda_const_cuda_0d(ExecutionContext* ctx,
                                             void* stream_handle,
                                             const char* unique_name);

template <typename T, int N>
MemRefType<T, N> ral_base_cuda_const_host(ExecutionContext* ctx,
                                          void* stream_handle,
                                          const char* unique_name);

template <typename T>
MemRefType<T, 0> ral_base_cuda_const_host_0d(ExecutionContext* ctx,
                                             void* stream_handle,
                                             const char* unique_name);

static inline void* aligned_malloc(int64_t size) {
  int align = ALIGN_BYTES;
  void* void_ptr = nullptr;
  posix_memalign(&void_ptr, align, size);
  return void_ptr;
}

buffer_shape_t GetShapeFromConstUniqueName(ExecutionContext* ctx,
                                           const std::string& unique_name,
                                           int64_t* width_in_bytes);

std::vector<uint8_t> fromHex(const std::string& Input);

// Returns ture if in debug mode.
bool isDebugMode();

// Returns true is profling mode is enabled.
bool isProfilingEnabled();

// Returns the maximum number of cpu cores DISC can uses.
int getNumAvailableCores();

// RAII object for timing measure
struct CpuTimer {
  // Starts the time at construction time.
  CpuTimer(const char* message);

  // If enabled, print the message and the time elapsed when destroyed.
  ~CpuTimer();

  // Explicitly stop the timer
  void Stop();

  // When `Stop` is called before, this return the elapsed time in `ns`.
  size_t GetNanoSeconds();

  bool stopped = false;
  const char* message;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point finish;
  size_t nanoseconds = 0;
};

}  // namespace ral
}  // namespace tao

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_COMMON_CONTEXT_IMPL_H_
