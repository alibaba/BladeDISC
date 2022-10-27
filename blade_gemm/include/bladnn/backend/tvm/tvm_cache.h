#pragma once

#include <bladnn/bladnn.h>
#include <bladnn/utils/log.h>
#include <hip/hip_runtime.h>
#include <pthread.h>

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "bladnn/backend/tvm/tvm_collector.h"
#include "bladnn/utils/common.h"

namespace bladnn {
namespace tvm {

static constexpr const int kMaxNumGPUs = 32;
static constexpr const int kMaxWorkSpace = 8388608;
static constexpr const int kMemAlignment = 256;
static constexpr const char* kTVMVersion = "0.1.0";
static constexpr const char* kDefaultMetaFuncName = "default_function_kernel0";
static constexpr const char* kUseDynamicSharedMemoryTag =
    "tir.use_dyn_shared_memory";

enum TVMGemmTranspose : int { NoTranspose = 0, Transpose = 1 };

enum TVMFuncCacheStatus { Hit, Miss, Invalid };

typedef std::string TVMFuncDataType;

struct TVMKernelInfo {
  std::string name;
  std::string kernel_name;
  std::vector<TVMFuncDataType> arg_types;
  std::vector<std::string> thread_axis_tags;
  std::vector<uint64_t> thread_axis_args;
};

struct TVMStorageInfo {
  std::string name;
  size_t size;
  std::string dtype;
};

struct TVMFuncInfo {
  std::vector<TVMKernelInfo> kernel;
  std::vector<TVMStorageInfo> storage;
};

struct ThreadScope {
  int rank{0};
  int dim_index{0};

  // Convert 'x', 'y', 'z' to idex 0, 1, 2
  static int DimToIndex(const char& dim) {
    if (dim == 'x' || dim == 'y' || dim == 'z') {
      return static_cast<int>(dim - 'x');
    } else {
      BLADNN_LOG(FATAL) << "Unknown dimension " << dim << " in threadscope ";
    }
  }

  //  Create thread scopes from kernel's meta data, tag eg: "blockIdx.x",
  //  "blockIdx.y", "blockIdx.z",
  // "threadIdx.x", "threadIdx.y", "threadIdx.z".
  static ThreadScope Create(const std::string& tag) {
    ThreadScope r;
    if (utils::startswith(tag, "vthread") || tag == "cthread") {
      r.rank = 1;
      r.dim_index = -1;
    } else if (utils::startswith(tag, "blockIdx.")) {
      r.rank = 0;
      r.dim_index = DimToIndex(tag[9]);
    } else if (utils::startswith(tag, "threadIdx.")) {
      r.rank = 1;
      r.dim_index = DimToIndex(tag[10]);
    } else {
      BLADNN_LOG(FATAL) << "Unknown threadscope " << tag;
    }
    return r;
  }
};

struct ThreadWorkLoad {
  size_t work_size[6];
  size_t dyn_shmem_size{0};
  inline size_t block_dim(size_t i) const { return work_size[i + 3]; }
  inline size_t grid_dim(size_t i) const { return work_size[i]; }
  void FillWorksize() { std::fill(work_size, work_size + 6, 1); }
};

class LaunchParamConfig {
 public:
  LaunchParamConfig(){};
  LaunchParamConfig(size_t base,
                    const std::vector<std::string>& launch_param_tags);
  // extract workload from arguments.
  ThreadWorkLoad Extract(const std::vector<uint64_t>& args) const;
  size_t work_dim() const { return work_dim_; }

 private:
  size_t base_;
  size_t work_dim_;
  std::vector<uint32_t> arg_index_map_;
  bool use_dyn_shared_memory_{false};
};

class TVMWorkSpace {
 private:
  void* buffer_;
  size_t capacity_ = kMaxWorkSpace;
  size_t alignment_ = kMemAlignment;
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t used_ = 0;
  std::mutex mtx_;

 public:
  TVMWorkSpace();
  ~TVMWorkSpace();
  std::vector<void*> Allocate(const std::vector<TVMStorageInfo>& size);
  void Free();
};

class TVMFuncValue {
 private:
  /* data */
  const std::string data_;
  const std::string name_;
  const TVMFuncInfo info_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<std::vector<hipFunction_t>, kMaxNumGPUs> fcache_;
  mutable std::array<hipModule_t, kMaxNumGPUs> module_;
  mutable hipFunction_t func_;
  hipFunction_t GetFunc(int idx = 0) const;
  const TVMFuncCacheStatus status_;
  // launch parameters configuration
  mutable std::vector<LaunchParamConfig> launch_param_config_;
  mutable std::vector<ThreadWorkLoad> wl_;
  mutable pthread_rwlock_t lock_;
  TVMWorkSpace* workspace_;

 public:
  TVMFuncValue() : status_(TVMFuncCacheStatus::Invalid){};
  TVMFuncValue(const std::string& data, const TVMFuncInfo& info,
               const std::string& name, TVMWorkSpace* workspace);
  ~TVMFuncValue();
  bool LaunchKernel(hipFunction_t func, hipStream_t stream, void* packed_args,
                    size_t packed_nbytes, void** kernelParams, int idx) const;
  // template<typename T>
  bool Launch(void* stream, void* packed_args, size_t packed_nbytes,
              void** kernelParams = nullptr) const;
  const std::string& Name() const { return name_; }
  const TVMFuncCacheStatus& Status() const { return status_; }
  bool IsHit() const { return status_ == TVMFuncCacheStatus::Hit; }
};

class TVMFuncCache {
 private:
  std::unordered_map<std::string, TVMFuncValue> content_;
  bool initialized_ = false;
  std::string op_;
  std::string device_;
  mutable pthread_rwlock_t lock_;
  TVMWorkSpace workspace_;

 public:
  TVMFuncCache();
  TVMFuncCache(const std::string& op, const std::string& device);
  ~TVMFuncCache();
  bool Init();
  bool Initialized() const { return initialized_; };
  void Insert(const std::string& key, TVMFuncValue&& value,
              bool overwrite = true);
  const TVMFuncValue& LookUp(const std::string& key) const;
};

const TVMFuncValue CACHE_EMPTY;

class TVMFuncCaches {
 private:
  std::unordered_map<std::string, TVMFuncCache*> caches_;
  mutable pthread_rwlock_t lock_;

 public:
  TVMFuncCaches();
  ~TVMFuncCaches();
  TVMFuncCache* CreateOrGet(const std::string& op, const std::string& device);
  void ReInit();
};

}  // namespace tvm
}  // namespace bladnn
