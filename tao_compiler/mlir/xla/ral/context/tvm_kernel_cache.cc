

#include "tensorflow/compiler/mlir/xla/ral/context/tvm_kernel_cache.h"
#include "third_party/json/single_include/nlohmann/json.hpp"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "rocm/include/rocblas.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/stream_executor/rocm/rocm_blas.h"
// #include "tensorflow/stream_executor/rocm/rocm_blas.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <unordered_map>

#define ROCM_DRIVER_CALL(x)                                                                    \
  {                                                                                            \
    hipError_t result = x;                                                                     \
    if (result != hipSuccess && result != hipErrorDeinitialized) {                             \
      LOG(FATAL) << "ROCM HIP Error: " #x " failed with error: " << hipGetErrorString(result); \
    }                                                                                          \
  }

#define ROCM_CALL(func)                                              \
  {                                                                  \
    hipError_t e = (func);                                           \
    CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }



namespace tao {
namespace ral {
namespace tvm_impl {

using namespace ::stream_executor::gpu;

static TVMFuncCaches tvm_func_caches;

TVMFuncCache* TVMFuncCacheCreateOrGet(const std::string& op, const std::string& device) {
  return tvm_func_caches.CreateOrGet(op, device);
}
void TVMFuncCacheReInit() {
  tvm_func_caches.ReInit();
}

TVMFuncCache* TVMFuncCaches::CreateOrGet(const std::string& op, const std::string& device) {
  auto key = op + "_" + device;
  pthread_rwlock_rdlock(&lock_);
  auto it = caches_.find(key);
  if (it == caches_.end()) {
    pthread_rwlock_unlock(&lock_);
    auto cache = new TVMFuncCache(op, device);
    pthread_rwlock_wrlock(&lock_);
    caches_.emplace(std::make_pair(std::move(key), cache));
    pthread_rwlock_unlock(&lock_);
    return cache;
  }
  auto cache = it->second;
  pthread_rwlock_unlock(&lock_);
  return cache;
}

void TVMFuncCaches::ReInit() {
  pthread_rwlock_wrlock(&lock_);
  for (auto it : caches_) {
    auto cache = it.second;
    cache->Init();
  }
  pthread_rwlock_unlock(&lock_);
}

TVMFuncCaches::TVMFuncCaches() {
  CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
}

TVMFuncCaches::~TVMFuncCaches() {
  for (auto it : caches_) {
    auto cache = it.second;
    delete cache;
  }
  CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}


namespace {
  static tensorflow::int64 rocm_profile() {
  static bool checked = false;
  static tensorflow::int64 profile = 0;
  if (checked)  {
    return profile;
  }
  tensorflow::ReadInt64FromEnvVar("DISC_OPS_PROFILING", 0,
                                 &profile);
  checked = true;
  return profile;
}

}

template <typename NativeT>
inline std::string GetTVMFuncKey() {
  LOG(FATAL) << "Not supported OPT func key type.";
  return "";
}


template<>
inline std::string GetTVMFuncKey<float>() {
  return "float32";
}

template<>
inline std::string GetTVMFuncKey<double>() {
  return "float64";
}

template<typename InT, typename OutT, typename AlphaBeta> 
inline std::string GetGemmTVMFuncKey(
     const std::string& device,
     int64_t m, int64_t n, int64_t k,
     tvm_impl::TVMGemmTranspose trans_a, tvm_impl::TVMGemmTranspose trans_b) {

    // std::chrono::system_clock::time_point t0, t1, t2, t3 ,t4, t5;       
    // // VLOG(0) << "Start get func";   
    // if (rocm_profile() == 1) {
    //   t0 = std::chrono::system_clock::now();
    // }    
    //  VLOG(0) << "Inside get key.";  
    // std::stringstream ss;
    // std::string ss;// = "gemm_";

    char buffer[1024];

    // if (rocm_profile() == 1) {
    //   t1 = std::chrono::system_clock::now();
    // }   
    sprintf(buffer, "%s_gemm_%d_%d_%d_%d_%d_%d_%s_%s_%s_%s", 
      device.c_str(), m, n ,k, trans_a, trans_b,
      tvm_impl::TVMGemmTranspose::NoTranspose,
      GetTVMFuncKey<InT>().c_str() , 
      GetTVMFuncKey<InT>().c_str() , 
      GetTVMFuncKey<OutT>().c_str() , 
      GetTVMFuncKey<AlphaBeta>().c_str() 
    );

    // absl::StrAppend(&ss, device, "_", 
    // std::to_string(m) , "_", 
    // std::to_string(n) , "_", 
    // std::to_string(k) , "_", 
    // std::to_string(trans_a) , "_", 
    // std::to_string(trans_b) , "_", 
    // std::to_string(tvm_impl::TVMGemmTranspose::NoTranspose) , "_", 
    // GetTVMFuncKey<InT>() , "_", 
    // GetTVMFuncKey<InT>() , "_", 
    // GetTVMFuncKey<OutT>() , "_", 
    // GetTVMFuncKey<AlphaBeta>() 
    // );


    //  ss << "gemm" << "_"  << "dcu" << "_";
    //  ss <<  m << "_" << n << "_" << k << "_";
    //  ss << trans_a << "_" << trans_b << "_" << tvm_impl::TVMGemmTranspose::NoTranspose <<  "_";
    //  ss << GetTVMFuncKey<InT>() << "_" << GetTVMFuncKey<InT>() << "_";
    //  ss << GetTVMFuncKey<OutT>() << "_" << GetTVMFuncKey<AlphaBeta>();
    // //  VLOG(0) << "Finish get key inside " << ss.str(); 
    // if (rocm_profile() == 1) {
    //   t2 = std::chrono::system_clock::now();
    // }   
    auto ss = std::string(buffer);
  
    // auto s = ss.str();
    // if (rocm_profile() == 1) {
    //   t3 = std::chrono::system_clock::now();
    //   auto time0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    //   auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //   auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    //   VLOG(0) << "TimeDur for genrate " << 
    //      time0 << " " << time1 << " " << time2 << " us";
    // }

    return ss;
}

using namespace tao::ral::gpu;

namespace {
  void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}


bool LoadMetaDataFromFile(const std::string& file_name,
                          TVMFuncInfo& info) {
  std::ifstream fs(file_name.c_str());
  CHECK(!fs.fail()) << "Cannot open file " << file_name;
  nlohmann::json meta = nlohmann::json::parse(fs);
  CHECK(meta["tvm_version"] == kTVMVersion) << "OPT version mismatch " << meta["tvm_version"] << " vs " << std::string(kTVMVersion);
  info.kernel.resize(meta["func_info"].size());
  for (auto& it : meta["func_info"].items()) {
    auto& key = it.key();
    int idx = key.back() - '0';
    info.kernel[idx].kernel_name = key;
    for (auto& item : it.value()["arg_types"]) {
      info.kernel[idx].arg_types.emplace_back(item);
    }
    for (auto& item : it.value()["launch_param_tags"]) {
      info.kernel[idx].thread_axis_tags.emplace_back(item);
    }
    for (auto& item : it.value()["launch_param_args"]) {
      info.kernel[idx].thread_axis_args.emplace_back(item);
    }
    info.kernel[idx].name = std::move(it.value()["name"]);
  }
  info.storage.resize(meta["storage"].size());
  for (auto& it : meta["storage"].items()) {
      auto& name = it.key();
      int idx = name.back() - '0';
      if (idx < 0 || idx > 9) {
        idx = 0;
      }
      info.storage[idx].name = name;
      info.storage[idx].dtype = it.value()[0];
      info.storage[idx].size =  std::stoll(std::string(it.value()[1]), nullptr, 10);
      if (info.storage[idx].dtype == "float64") {
        info.storage[idx].size  *= 8;
      } else if (info.storage[idx].dtype == "float32") {
        info.storage[idx].size  *= 4;
      }
  }
  fs.close();
  return true;
}

}

TVMFuncValue::TVMFuncValue(const std::string& data, const TVMFuncInfo& info, const std::string& name) : 
   data_(data), info_(info), name_(name), status_(TVMFuncCacheStatus::Hit) {
  for (auto& i : info.kernel) {
    launch_param_config_.emplace_back(LaunchParamConfig(i.arg_types.size(), i.thread_axis_tags));
    wl_.emplace_back(launch_param_config_.back().Extract(i.thread_axis_args));
  }
  // VLOG(0) << "TVM func value fill here.";
  module_.fill(nullptr);
  fcache_.fill(std::vector<hipFunction_t>(info.kernel.size(), nullptr));
  // flag_ = -1;
  CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
};

// TODO(fl237079): Add clear stuffs
TVMFuncValue::~TVMFuncValue()
{
   CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}
  
hipFunction_t TVMFuncValue::GetFunc(int idx) const {
    int device_id;
    // VLOG(0) << "Before get device inside.";
    ROCM_CALL(hipGetDevice(&device_id));
    // VLOG(0) << "Finish get device inside.";
    pthread_rwlock_rdlock(&lock_);
    // VLOG(0) << "flag start " << flag_;
    // if (fcache_[device_id] != nullptr) {
    //   VLOG(0) << "Debug start not null for " << device_id; 
    // } else {
    //   VLOG(0) << "Debug start null for " << device_id; 
    // }
    // if (func_ != nullptr) {
    //   VLOG(0) << "Debug start func not null for " << device_id; 
    // } else {
    //    VLOG(0) << "Debug start func null for " << device_id; 
    // }

    // VLOG(0) << "value ptr" << this;
    if (fcache_[device_id][idx] != nullptr) {
      // VLOG(0) << "Function cache already hit.";
      auto func = fcache_[device_id][idx];
      pthread_rwlock_unlock(&lock_);
      return func;
    }
    pthread_rwlock_unlock(&lock_);
    pthread_rwlock_wrlock(&lock_);
    // must recheck under the lock scope
    // VLOG(0) << "Get module size are " << data_.size() << " " << data_.length() << " ";
    // VLOG(0) << data_;
    if (module_[device_id] == nullptr) {
      VLOG(1) << "Get module for device " << device_id;
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
   
    hipFunction_t func;
    auto& func_name = info_.kernel[idx].kernel_name;
    VLOG(1) << "Get func " << func_name << "for device " << device_id;
    hipError_t result = hipModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != hipSuccess) {
      LOG(FATAL) << "ROCMError: hipModuleGetFunction " << func_name
                 << " failed with error: " << hipGetErrorString(result);
    }
    fcache_[device_id][idx] = func;
    // func_ = func;
    // flag_ = 1;
    // VLOG(0) << "flag " << flag_;
    // if (fcache_[device_id] != nullptr) {
    //   VLOG(0) << "Debug not null for " << device_id; 
    // } else {
    //    VLOG(0) << "Debug null for " << device_id; 
    // }
    // if (func_ != nullptr) {
    //   VLOG(0) << "Debug func not null for " << device_id; 
    // } else {
    //    VLOG(0) << "Debug func null for " << device_id; 
    // }
    pthread_rwlock_unlock(&lock_);
    return func;
}

bool TVMFuncValue::Launch(unsigned int gridDimX,
                unsigned int gridDimY, unsigned int gridDimZ,
                unsigned int blockDimX, unsigned int blockDimY,
                unsigned int blockDimZ, unsigned int sharedMemBytes,
                se::Stream* stream, void** kernelParams,
                void** extra) const {
    hipFunction_t func = GetFunc(); 
    hipStream_t strm = stream ? AsGpuStreamValue(stream) : nullptr;
    ROCM_DRIVER_CALL(hipModuleLaunchKernel(func, gridDimX,
                gridDimY, gridDimZ,
                blockDimX, blockDimY,
                blockDimZ, sharedMemBytes,
                                          strm, kernelParams,
                                           extra));
    return true;
}

bool TVMFuncValue::LaunchKernel(hipFunction_t func, hipStream_t strm, void* packed_args, 
        size_t packed_nbytes, void** kernelParams, int idx) const {
          
   void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, packed_args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &packed_nbytes, HIP_LAUNCH_PARAM_END};
    // VLOG(0) << "Launch tvm kernel for " << name_;
    uint64_t* deb = static_cast<uint64_t*>(packed_args);
    VLOG(1) << "OPT kernel launch params " << wl_[idx].grid_dim(0) << " " << wl_[idx].grid_dim(1) <<
              " " <<  wl_[idx].grid_dim(2) << " " << wl_[idx].block_dim(0) << " " << wl_[idx].block_dim(1)
              <<  " " <<  wl_[idx].block_dim(2) << " " << wl_[idx].dyn_shmem_size;
    VLOG(1) << "OPT kernel packed args num " << packed_nbytes << " : " << deb[0] << " " << deb[1] << 
       " " <<  deb[2];
    // VLOG(0) << config[0] << " " << config[1] << " " << config[2] << " " << config[3] <<  " " << config[4] << std::endl;  
    // VLOG(0) << func << " " << strm << " " << kernelParams << " " << static_cast<void**>(config);
    se::port::Status block_status;
    // if (rocm_profile() == 1) {
    //   t2 = std::chrono::system_clock::now();
    //   block_status = stream->BlockHostUntilDone();
    //   t3 = std::chrono::system_clock::now();
    // }
    hipEvent_t start, stop;
    if (rocm_profile() == 2) {
      hipEventCreate(&start);
      hipEventCreate(&stop);
      hipEventRecord(start, strm);
    } 
    ROCM_DRIVER_CALL(hipModuleLaunchKernel(func, wl_[idx].grid_dim(0), wl_[idx].grid_dim(1),
                wl_[idx].grid_dim(2), wl_[idx].block_dim(0), wl_[idx].block_dim(1),
                wl_[idx].block_dim(2), wl_[idx].dyn_shmem_size,
                strm, kernelParams, reinterpret_cast<void**>(&config)));

    // if (rocm_profile() == 1) {
    //   t4 = std::chrono::system_clock::now();
    //   block_status = stream->BlockHostUntilDone();
    //   t5 = std::chrono::system_clock::now();
    //   auto time0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    //   auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    //   auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    //   auto time3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    //   auto time4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

    //   VLOG(0) << "TimeDur for kernel " << name_ << " " << 
    //      time0 << " " << time1 << " " << time2 << " " << time3 
    //      << " " << time4 << " us";

    //   // auto time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t3).count();   
    //   // VLOG(0) << "TimeDur for kernel " << name_ << " " << 
    //   //    time << " us";
    // }


   if (rocm_profile() == 2) {
    hipEventRecord(stop, strm);
    hipEventSynchronize(stop);
    float eventMs;
    hipEventElapsedTime(&eventMs, start, stop);
    VLOG(0) << "TimeDur event " << name_ << " : " << eventMs * 1000 << " us";
  } 

  
                // kernelParams, 
                // static_cast<void**>(&config));

    // double alpha = 1.0;
    // double beta = 0.0; 
    // se::blas::BlasSupport *blas = stream->parent()->AsBlas();
    // blas->DoBlasGemm(stream, se::blas::Transpose::kNoTranspose,
    //                       se::blas::Transpose::kTranspose, 100 , 50 , 11776,
    //                       alpha, a, 100,
    //                       b, 50, beta,
    //                       c, 100);    

 
    // ROCMBlas::DoBlasInternal(
    //   wrap::rocblas_dgemm, strm, true /* = pointer_mode_host */,
    //   se::blas::Transpose::kNoTranspose, se::blas::Transpose::kTranspose, 100, 50, 11776, &alpha,
    //   deb[1], 100, deb[0], 50, &beta, deb[2], 100);                      

    VLOG(1) << "Finish call tvm kernel for " << name_ << " " << idx; 
    return true;            
}




// template<typename T>
bool TVMFuncValue::Launch(se::Stream* stream, void* packed_args, 
        size_t packed_nbytes, se::ScratchAllocator *scratch_allocator, void** kernelParams
        // const se::DeviceMemory<T> &a,
        // const se::DeviceMemory<T> &b,
        // se::DeviceMemory<T> *c
        ) const {
    hipStream_t strm = stream ? AsGpuStreamValue(stream) : nullptr;  
    auto kernel_cnt = info_.kernel.size();
    void** input_args = static_cast<void**>(packed_args); 
    std::vector<se::DeviceMemory<se::uint8>> scratch_buffers;
    auto scratch_alloc = [&](int idx) {
        VLOG(1) << "Scratch DISC opt alloc size " << info_.storage[idx].size;
        auto allocated = scratch_allocator->AllocateBytes(info_.storage[idx].size);
        if (allocated.ok()) {
          scratch_buffers.emplace_back(allocated.ValueOrDie());
        } else {
          LOG(FATAL) << "Alloc Scratch error " << name_ << " at idx " << idx;
        }
    };
    hipEvent_t start, stop;
    if (rocm_profile() == 3) {
      hipEventCreate(&start);
      hipEventCreate(&stop);
      hipEventRecord(start, strm);
    } 
    for (int idx = 0; idx < info_.kernel.size(); idx++) {
      hipFunction_t func = GetFunc(idx); 
      void* args;
      std::vector<void*> args_content;
      if (kernel_cnt == 1) {
        args = packed_args;
      } else if (kernel_cnt == 2) {
        if (idx == 0) {
          scratch_alloc(0);
          args_content = {input_args[0], input_args[1], scratch_buffers[0].opaque()};
          args = static_cast<void*>(args_content.data());
        } else if (idx == 1) {
          args_content = {scratch_buffers[0].opaque(), input_args[2]};
          args = static_cast<void*>(args_content.data());          
        }
      } else if (kernel_cnt == 3) {
        if (idx == 0) {
          scratch_alloc(0);
          args_content = {input_args[0], scratch_buffers[0].opaque()};
          args = static_cast<void*>(args_content.data());
        } else if (idx == 1) {
          scratch_alloc(1);
          args_content = {input_args[1], scratch_buffers[1].opaque()};
          args = static_cast<void*>(args_content.data());          
        } else if (idx == 2) {
          args_content = {scratch_buffers[0].opaque(), scratch_buffers[1].opaque(), input_args[2]};
          args = static_cast<void*>(args_content.data()); 
        }
      }
      if (!LaunchKernel(func, strm, args, 
          sizeof(args_content), nullptr, idx)) {
         LOG(FATAL) << "Launch OPT kernel error for kernel " << name_ << " at idx " << idx;
      }
    }
    if (rocm_profile() == 3) {
      hipEventRecord(stop, strm);
      hipEventSynchronize(stop);
      float eventMs;
      hipEventElapsedTime(&eventMs, start, stop);
      VLOG(0) << "TimeDur event " << name_ << " : " << eventMs * 1000 << " us";
    }                    
    VLOG(1) << "Finish call tvm func for " << name_; 
    return true;            
}

TVMFuncCache::TVMFuncCache() {
  CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
}

TVMFuncCache::TVMFuncCache(const std::string& op, const std::string& device) {
  CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
  op_ = op;
  device_ = device;
  Init();
}

TVMFuncCache::~TVMFuncCache() {
  CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}

void TVMFuncCache::Insert(const std::string& key, TVMFuncValue&& value, bool overwrite) { 
  // pthread_rwlock_wrlock(&lock_);
  if (content_.find(key) == content_.end() || overwrite) {
    content_.emplace(key, std::move(value));
  } 
  // pthread_rwlock_unlock(&lock_);
}

const TVMFuncValue& TVMFuncCache::LookUp(const std::string& key) const {
  pthread_rwlock_rdlock(&lock_);
  if (content_.find(key) != content_.end()) {
    const TVMFuncValue& func = content_.at(key);
    pthread_rwlock_unlock(&lock_);
    // VLOG(0) << "Look up " << func.flag_;
    return func;
  }
  pthread_rwlock_unlock(&lock_); 
  return CACHE_EMPTY;
}

bool TVMFuncCache::Init() {
    auto op = op_;
    auto device = device_;
    // VLOG(0) << "lock 0";
    pthread_rwlock_wrlock(&lock_);  
    // if (initialized_) {
    //   pthread_rwlock_unlock(&lock_);  
    //   return true;
    // }
    VLOG(1) << "Start init OPT func cache.";
    if (const char* local_lib_path = std::getenv("TAO_OPT_KERNEL_PATTERN_ROCM")) {
    if (auto* dir = opendir(local_lib_path)) {
      int count = 0;
      while (const auto* ent = readdir(dir)) {
        // VLOG(0) << ent->d_name;
        int len = strlen(ent->d_name);
        std::string prefix = device + "_" + op;
        if (strncmp(ent->d_name, prefix.c_str(), prefix.length()) != 0) {
          continue;
        }
        if (len > 6 && strcmp(ent->d_name + len - 6, ".hsaco") == 0) {
          // VLOG(0) << "Inside " << ent->d_name; 
          std::string key = ent->d_name;
          std::string so_path = local_lib_path[strlen(local_lib_path)] == '/'
                                    ? std::string(local_lib_path) + key
                                    : std::string(local_lib_path) + "/" + key;
          key.erase(key.end() - 6, key.end());
          std::string meta_path = local_lib_path[strlen(local_lib_path)] == '/'
                                    ? std::string(local_lib_path) + key + ".meta_for_tao.json"
                                    : std::string(local_lib_path) + "/" + key + ".meta_for_tao.json";
          std::string data;
          TVMFuncInfo info;
          LoadBinaryFromFile(so_path, &data);
          CHECK(LoadMetaDataFromFile(meta_path, info)) << "Load meta date file error from " << meta_path;
          Insert(key, TVMFuncValue(data, info, key));
          VLOG(1) << "Finish insert func cache for " << key;
          count++;
        }
      }
      VLOG(1) << "OPT kernel impl " << count
                  << " kernel load from local lib: " << local_lib_path;
      closedir(dir);
      initialized_ = true;
      pthread_rwlock_unlock(&lock_);
      return true;
    }
  }
  LOG(WARNING) << "OPT local pattern lib not found.";
  initialized_ = true;
  pthread_rwlock_unlock(&lock_);  
  return false;
}


template
std::string GetGemmTVMFuncKey<float, float, float>(
     const std::string& device,
     int64_t m, int64_t n, int64_t k,
     tvm_impl::TVMGemmTranspose trans_a, tvm_impl::TVMGemmTranspose trans_b); 

template 
std::string GetGemmTVMFuncKey<double, double, double>(
     const std::string& device,
     int64_t m, int64_t n, int64_t k,
     tvm_impl::TVMGemmTranspose trans_a, tvm_impl::TVMGemmTranspose trans_b); 



template<>
std::string GetGemmTVMFuncKey<Eigen::half, Eigen::half, float>(
     const std::string& device,
     int64_t m, int64_t n, int64_t k,
     tvm_impl::TVMGemmTranspose trans_a, tvm_impl::TVMGemmTranspose trans_b) {
       LOG(FATAL) << "Data type not supported for half.";
    return "";
}; 




}  // namespace tvm_impl 
}  // namespace ral
}  // namespace tao
