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

#include <array>
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
  RalGlobalConstantState() {
    for (int i = 0; i < host_constants_by_idx.size(); ++i) {
      host_constants_by_idx[i] = nullptr;
      device_constants_by_idx[i] = nullptr;
    }
  }

  std::mutex mu;
  mlir::MetadataProto metadata_proto;
  // If not null, use the process level const store instead of this context
  // level store
  ProcessLevelConstStore* process_level_store = nullptr;
  // map <unique_name, <device/host_ptr, shape>>
  // in theory, there might be two constants with the same data value but
  // different shapes however for simplicity we just use the whole unique_name
  // as the key. This can be further optimized in case neccessary.
  using Item = std::pair<buffer_t, buffer_shape_t>;
  using ItemMap = std::unordered_map<std::string, Item>;
  ItemMap device_constants;
  ItemMap host_constants;

  // fast path: using a unique const index to do look up.
  // Note that the const index is assigned to each const at compile time.
  // The index is unique within the compiled module level.
  //
  // The large index suppored to use the fast path.
  static constexpr const int kMaxNumItemsForFastPath = 8196;
  template <typename T>
  using ItemArray = std::array<T, kMaxNumItemsForFastPath>;
  using ItemLookupFastPathStorage = ItemArray<Item>;
  using ItemLookupFastPathTable = ItemArray<std::atomic<Item*>>;

  // for host const
  ItemLookupFastPathStorage host_constants_storage;
  ItemLookupFastPathTable host_constants_by_idx;
  // for device const
  ItemLookupFastPathStorage device_constants_storage;
  ItemLookupFastPathTable device_constants_by_idx;

  // Returns nullptr if not found or supported.
  Item* getHostConstByIndex(int unique_index_in_module) {
    if (unique_index_in_module >= host_constants_by_idx.size()) return nullptr;
    return host_constants_by_idx[unique_index_in_module];
  }
  // update the item according to `unique_index_in_module`
  void setHostConstByIndex(int unique_index_in_module, Item item) {
    if (unique_index_in_module >= host_constants_by_idx.size()) return;
    host_constants_storage[unique_index_in_module] = std::move(item);
    host_constants_by_idx[unique_index_in_module] =
        &host_constants_storage[unique_index_in_module];
  }

  // Returns nullptr if not found or supported.
  Item* getDeviceConstByIndex(int unique_index_in_module) {
    if (unique_index_in_module >= device_constants_by_idx.size())
      return nullptr;
    return device_constants_by_idx[unique_index_in_module];
  }
  // update the item according to `unique_index_in_module`
  void setDeviceConstByIndex(int unique_index_in_module, Item item) {
    if (unique_index_in_module >= device_constants_by_idx.size()) return;
    device_constants_storage[unique_index_in_module] = std::move(item);
    device_constants_by_idx[unique_index_in_module] =
        &device_constants_storage[unique_index_in_module];
  }

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
                                          const char* unique_name,
                                          int32_t unique_index_in_module);

template <typename T>
MemRefType<T, 0> ral_base_cuda_const_cuda_0d(ExecutionContext* ctx,
                                             void* stream_handle,
                                             const char* unique_name,
                                             int32_t unique_index_in_module);

template <typename T, int N>
MemRefType<T, N> ral_base_cuda_const_host(ExecutionContext* ctx,
                                          void* stream_handle,
                                          const char* unique_name,
                                          int32_t unique_index_in_module);

template <typename T>
MemRefType<T, 0> ral_base_cuda_const_host_0d(ExecutionContext* ctx,
                                             void* stream_handle,
                                             const char* unique_name,
                                             int32_t unique_index_in_module);

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
