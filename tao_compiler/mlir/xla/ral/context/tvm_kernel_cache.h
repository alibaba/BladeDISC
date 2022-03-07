#ifndef RAL_CONTEXT_TVM_KERNEL_IMPL_H_
#define RAL_CONTEXT_TVM_KERNEL_IMPL_H_

#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"
#include <map>
#include <pthread.h>

// #ifdef TENSORFLOW_USE_DCU
// #define TVM_GPU_DEVICE "dcu"
// #else 
// #ifdef TENSORFLOW_USE_ROCM
// #define TVM_GPU_DEVICE "rocm"
// #endif
// #endif



namespace tao {
namespace ral {
namespace tvm_impl {

namespace se = ::stream_executor;

enum TVMGemmTranspose: int {
  NoTranspose=0,
  Transpose=1
};

static constexpr const int kMaxNumGPUs = 32;
static constexpr const char* kTVMVersion = "0.1.0";
static constexpr const char* kDefaultMetaFuncName = "default_function_kernel0";
static constexpr const char* kUseDynamicSharedMemoryTag = "tir.use_dyn_shared_memory";

enum TVMFuncCacheStatus {
    Hit,
    Miss,
    Invalid
};

// enum TVMFuncDataType {
//     Handle,
//     Int32,
//     Int64
// };

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
  static ThreadScope Create(const std::string& s) {
    ThreadScope r;
    if (s.compare(0, 7, "vthread") == 0 || s == "cthread") {
      r.rank = 1;
      r.dim_index = -1;
    } else if (s.compare(0, 9, "blockIdx.") == 0) {
      r.rank = 0;
      r.dim_index = static_cast<int>(s[9] - 'x');
    } else if (s.compare(0, 10, "threadIdx.") == 0) {
      r.rank = 1;
      r.dim_index = static_cast<int>(s[10] - 'x');
    } else {
      LOG(FATAL) << "Unknown threadscope " << s;
    }
    return r;
  }
};


struct ThreadWorkLoad {
  size_t work_size[6];
  size_t dyn_shmem_size{0};
  inline size_t block_dim(size_t i) const { return work_size[i + 3]; }
  inline size_t grid_dim(size_t i) const { return work_size[i]; }
};


class LaunchParamConfig {
 public:
  LaunchParamConfig() {};
  LaunchParamConfig(size_t base, const std::vector<std::string>& launch_param_tags) {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      const std::string& tag = launch_param_tags[i];
      if (tag == kUseDynamicSharedMemoryTag) {
        CHECK(i == launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
      } else {
        ThreadScope ts = ThreadScope::Create(tag);
        arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
        filled[ts.rank * 3 + ts.dim_index] = true;
      }
    }
    work_dim_ = 1;
    for (int i = 0; i < 3; ++i) {
      if (filled[i] || filled[i + 3]) {
        work_dim_ = i + 1;
      }
    }
  }
  // extract workload from arguments.
  ThreadWorkLoad Extract(const std::vector<uint64_t>& args) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      size_t size = static_cast<size_t>(args[i]);
      if (size > 0) {
        w.work_size[arg_index_map_[i]] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(args[arg_index_map_.size()]);
    }
    return w;
  }
  size_t work_dim() const { return work_dim_; }
 private:
  size_t base_;
  size_t work_dim_;
  std::vector<uint32_t> arg_index_map_;
  bool use_dyn_shared_memory_{false};
};

template<typename InT, typename OutT, typename AlphaBeta> 
inline std::string GetGemmTVMFuncKey(const std::string& device,
    int64_t m, int64_t n, int64_t k,
     tvm_impl::TVMGemmTranspose trans_a, tvm_impl::TVMGemmTranspose trans_b);


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
    // launch parameters configuration
    hipFunction_t GetFunc(int idx = 0) const;
    // unsigned int gridDimX_;
    // unsigned int gridDimY_; 
    // unsigned int gridDimZ_;
    // unsigned int blockDimX_; 
    // unsigned int blockDimY_;
    // unsigned int blockDimZ_; 
    // unsigned int sharedMemBytes_;  
    const TVMFuncCacheStatus status_;
    mutable std::vector<LaunchParamConfig> launch_param_config_;
    mutable std::vector<ThreadWorkLoad> wl_;
    mutable pthread_rwlock_t lock_;


public:
    // mutable int flag_;
    TVMFuncValue() : status_(TVMFuncCacheStatus::Invalid) {};
    TVMFuncValue(const std::string& data, const TVMFuncInfo& info, const std::string& name);
    ~TVMFuncValue();
    bool LaunchKernel(hipFunction_t func, hipStream_t stream, void* packed_args, 
        size_t packed_nbytes, void** kernelParams, int idx) const; 
    bool Launch(unsigned int gridDimX,
                unsigned int gridDimY, unsigned int gridDimZ,
                unsigned int blockDimX, unsigned int blockDimY,
                unsigned int blockDimZ, unsigned int sharedMemBytes,
                se::Stream* stream, void** extra, void** kernelParams=nullptr) const; 
    // template<typename T>
    bool Launch(se::Stream* stream, void* packed_args, 
        size_t packed_nbytes,  se::ScratchAllocator *scratch_allocator, void** kernelParams=nullptr
        // ,
        // const se::DeviceMemory<T> &a,
        // const se::DeviceMemory<T> &b,
        // se::DeviceMemory<T> *c
        ) const;
    const std::string& Name() const {
        return name_;
    } 
    const TVMFuncCacheStatus& Status() const {
        return status_;
    }    
    bool IsHit() const {
        return status_ == TVMFuncCacheStatus::Hit;
    }
};


class TVMFuncCache {
private:
    std::unordered_map<std::string, TVMFuncValue> content_;
    bool initialized_ = false;
    std::string op_;
    std::string device_;
    mutable pthread_rwlock_t lock_;
public:
    TVMFuncCache();
    TVMFuncCache(const std::string& op, const std::string& device);
    ~TVMFuncCache();
    bool Init();
    bool Initialized() const {
        return initialized_;
    };
    void Insert(const std::string& key, TVMFuncValue&& value, bool overwrite=true);
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

TVMFuncCache* TVMFuncCacheCreateOrGet(const std::string& op, const std::string& device);

void TVMFuncCacheReInit();

}  // namespace tvm_impl 
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_TVM_KERNEL_IMPL_H_
