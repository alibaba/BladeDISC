#include <dirent.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "bladnn/backend/tvm/tvm_handle.h"
#include "bladnn/utils/env.h"
#include "bladnn/utils/log.h"
#include "hip/hip_runtime.h"
#include "nlohmann/json.hpp"

#define ROCM_DRIVER_CALL(x)                                             \
  {                                                                     \
    hipError_t result = x;                                              \
    if (result != hipSuccess && result != hipErrorDeinitialized) {      \
      BLADNN_LOG(FATAL) << "ROCM HIP Error: " #x " failed with error: " \
                        << hipGetErrorString(result);                   \
    }                                                                   \
  }

#define ROCM_CALL(func)                                                    \
  {                                                                        \
    hipError_t e = (func);                                                 \
    BLADNN_CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }

namespace bladnn {
namespace tvm {

TVMFuncCache* TVMFuncCaches::CreateOrGet(const std::string& op,
                                         const std::string& device) {
  auto key = op + "_" + device;
  pthread_rwlock_rdlock(&lock_);
  auto it = caches_.find(key);
  if (it == caches_.end()) {
    pthread_rwlock_unlock(&lock_);
    pthread_rwlock_wrlock(&lock_);
    TVMFuncCache* cache;
    if (caches_.count(key) <= 0) {
      cache = new TVMFuncCache(op, device);
      caches_.emplace(std::make_pair(std::move(key), cache));
    } else {
      cache = caches_.at(key);
    }
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
  BLADNN_CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
}

TVMFuncCaches::~TVMFuncCaches() {
  for (auto it : caches_) {
    auto cache = it.second;
    delete cache;
  }
  BLADNN_CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}

namespace {
static int rocmProfile() {
  static bool checked = false;
  static int profile = 0;
  if (checked) {
    return profile;
  }
  std::string str;
  utils::ReadStringFromEnvVar("BLADNN_OPS_PROFILING", "0", &str);
  profile = std::stoi(str);
  checked = true;
  return profile;
}
}  // namespace

namespace {
void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  BLADNN_CHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(const_cast<char*>(data->data()), size);
}

bool LoadMetaDataFromFile(const std::string& file_name, TVMFuncInfo& info) {
  std::ifstream fs(file_name.c_str());
  BLADNN_CHECK(!fs.fail()) << "Cannot open file " << file_name;
  nlohmann::json meta = nlohmann::json::parse(fs);
  BLADNN_CHECK(meta["tvm_version"] == kTVMVersion)
      << "OPT version mismatch " << meta["tvm_version"] << " vs "
      << kTVMVersion;
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
    info.storage[idx].size =
        std::stoll(std::string(it.value()[1]), nullptr, 10);
    if (info.storage[idx].dtype == "float64") {
      info.storage[idx].size *= 8;
    } else if (info.storage[idx].dtype == "float32") {
      info.storage[idx].size *= 4;
    }
  }
  fs.close();
  return true;
}

}  // namespace

TVMFuncValue::TVMFuncValue(const std::string& data, const TVMFuncInfo& info,
                           const std::string& name, TVMWorkSpace* workspace)
    : data_(data), info_(info), name_(name), status_(TVMFuncCacheStatus::Hit) {
  for (auto& i : info.kernel) {
    launch_param_config_.emplace_back(
        LaunchParamConfig(i.arg_types.size(), i.thread_axis_tags));
    wl_.emplace_back(launch_param_config_.back().Extract(i.thread_axis_args));
  }
  module_.fill(nullptr);
  fcache_.fill(std::vector<hipFunction_t>(info.kernel.size(), nullptr));
  workspace_ = workspace;
  BLADNN_CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
};

// TODO(fl237079): Add clear stuffs
TVMFuncValue::~TVMFuncValue() {
  BLADNN_CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}

hipFunction_t TVMFuncValue::GetFunc(int idx) const {
  int device_id;
  ROCM_CALL(hipGetDevice(&device_id));
  pthread_rwlock_rdlock(&lock_);
  if (fcache_[device_id][idx] != nullptr) {
    // BLADNN_VLOG(0) << "Function cache already hit.";
    auto func = fcache_[device_id][idx];
    pthread_rwlock_unlock(&lock_);
    return func;
  }
  pthread_rwlock_unlock(&lock_);
  pthread_rwlock_wrlock(&lock_);
  hipFunction_t func;
  if (fcache_[device_id][idx] != nullptr) {
    func = fcache_[device_id][idx];
  } else {
    if (module_[device_id] == nullptr) {
      BLADNN_VLOG(1) << "Get module for device " << device_id;
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    auto& func_name = info_.kernel[idx].kernel_name;
    BLADNN_VLOG(1) << "Get func " << func_name << "for device " << device_id;
    hipError_t result =
        hipModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != hipSuccess) {
      BLADNN_LOG(FATAL) << "ROCMError: hipModuleGetFunction " << func_name
                        << " failed with error: " << hipGetErrorString(result);
    }
    fcache_[device_id][idx] = func;
  }
  pthread_rwlock_unlock(&lock_);
  return func;
}

bool TVMFuncValue::LaunchKernel(hipFunction_t func, hipStream_t strm,
                                void* packed_args, size_t packed_nbytes,
                                void** kernelParams, int idx) const {
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, packed_args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &packed_nbytes,
                    HIP_LAUNCH_PARAM_END};
  // BLADNN_VLOG(0) << "Launch tvm kernel for " << name_;
  uint64_t* deb = static_cast<uint64_t*>(packed_args);
  BLADNN_VLOG(1) << "OPT kernel launch params " << wl_[idx].grid_dim(0) << " "
                 << wl_[idx].grid_dim(1) << " " << wl_[idx].grid_dim(2) << " "
                 << wl_[idx].block_dim(0) << " " << wl_[idx].block_dim(1) << " "
                 << wl_[idx].block_dim(2) << " " << wl_[idx].dyn_shmem_size;
  BLADNN_VLOG(1) << "OPT kernel packed args num " << packed_nbytes << " : "
                 << deb[0] << " " << deb[1] << " " << deb[2];
  hipEvent_t start, stop;
  if (rocmProfile() == 1) {
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, strm);
  }
  ROCM_DRIVER_CALL(hipModuleLaunchKernel(
      func, wl_[idx].grid_dim(0), wl_[idx].grid_dim(1), wl_[idx].grid_dim(2),
      wl_[idx].block_dim(0), wl_[idx].block_dim(1), wl_[idx].block_dim(2),
      wl_[idx].dyn_shmem_size, strm, kernelParams,
      reinterpret_cast<void**>(&config)));

  if (rocmProfile() == 1) {
    hipEventRecord(stop, strm);
    hipEventSynchronize(stop);
    float eventMs;
    hipEventElapsedTime(&eventMs, start, stop);
    BLADNN_VLOG(0) << "TimeDur for " << name_ << " : " << eventMs * 1000
                   << " us";
  }

  BLADNN_VLOG(1) << "Finish call optimized kernel for " << name_ << " " << idx;
  return true;
}

bool TVMFuncValue::Launch(void* stream, void* packed_args, size_t packed_nbytes,
                          void** kernelParams) const {
  hipStream_t strm = reinterpret_cast<hipStream_t>(stream);
  auto kernel_cnt = info_.kernel.size();
  void** input_args = static_cast<void**>(packed_args);
  std::vector<void*> scratch_buffers = workspace_->Allocate(info_.storage);
  if (scratch_buffers.empty() && !info_.storage.empty()) {
    workspace_->Free();
    return false;
  }
  hipEvent_t start, stop;
  if (rocmProfile() == 2) {
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
        args_content = {input_args[0], input_args[1], scratch_buffers[0]};
        args = static_cast<void*>(args_content.data());
      } else if (idx == 1) {
        args_content = {scratch_buffers[0], input_args[2]};
        args = static_cast<void*>(args_content.data());
      }
    } else if (kernel_cnt == 3) {
      if (idx == 0) {
        args_content = {input_args[0], scratch_buffers[0]};
        args = static_cast<void*>(args_content.data());
      } else if (idx == 1) {
        args_content = {input_args[1], scratch_buffers[1]};
        args = static_cast<void*>(args_content.data());
      } else if (idx == 2) {
        args_content = {scratch_buffers[0], scratch_buffers[1], input_args[2]};
        args = static_cast<void*>(args_content.data());
      }
    }
    if (!LaunchKernel(func, strm, args, sizeof(args_content), nullptr, idx)) {
      BLADNN_LOG(FATAL) << "Launch OPT kernel error for kernel " << name_
                        << " at idx " << idx;
    }
  }
  if (rocmProfile() == 2) {
    hipEventRecord(stop, strm);
    hipEventSynchronize(stop);
    float eventMs;
    hipEventElapsedTime(&eventMs, start, stop);
    BLADNN_VLOG(0) << "TimeDur with prepare for " << name_ << " : "
                   << eventMs * 1000 << " us";
  }
  workspace_->Free();
  BLADNN_VLOG(1) << "Finish call optimized func for " << name_;
  return true;
}

TVMFuncCache::TVMFuncCache() {
  BLADNN_CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
}

TVMFuncCache::TVMFuncCache(const std::string& op, const std::string& device) {
  BLADNN_CHECK(!pthread_rwlock_init(&lock_, NULL)) << "Read-write init error";
  op_ = op;
  device_ = device;
  Init();
}

TVMFuncCache::~TVMFuncCache() {
  BLADNN_CHECK(!pthread_rwlock_destroy(&lock_)) << "Read-write destroy error";
}

void TVMFuncCache::Insert(const std::string& key, TVMFuncValue&& value,
                          bool overwrite) {
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
    return func;
  }
  pthread_rwlock_unlock(&lock_);
  return CACHE_EMPTY;
}

bool TVMFuncCache::Init() {
  auto op = op_;
  auto device = device_;
  pthread_rwlock_wrlock(&lock_);
  BLADNN_VLOG(1) << "Start init tuned func cache.";
  if (const char* local_lib_path = std::getenv("BLADNN_KERNEL_CACHE")) {
    if (auto* dir = opendir(local_lib_path)) {
      int count = 0;
      while (const auto* ent = readdir(dir)) {
        int len = strlen(ent->d_name);
        std::string prefix = device + "_" + op;
        if (strncmp(ent->d_name, prefix.c_str(), prefix.length()) != 0) {
          continue;
        }
        if (len > 6 && strcmp(ent->d_name + len - 6, ".hsaco") == 0) {
          std::string key = ent->d_name;
          std::string so_path = local_lib_path[strlen(local_lib_path)] == '/'
                                    ? std::string(local_lib_path) + key
                                    : std::string(local_lib_path) + "/" + key;
          key.erase(key.end() - 6, key.end());
          std::string meta_path =
              local_lib_path[strlen(local_lib_path)] == '/'
                  ? std::string(local_lib_path) + key + ".meta_for_tao.json"
                  : std::string(local_lib_path) + "/" + key +
                        ".meta_for_tao.json";
          std::string data;
          TVMFuncInfo info;
          LoadBinaryFromFile(so_path, &data);
          BLADNN_CHECK(LoadMetaDataFromFile(meta_path, info))
              << "Load meta date file error from " << meta_path;
          Insert(key, TVMFuncValue(data, info, key, &workspace_));
          BLADNN_VLOG(1) << "Finish insert func cache for " << key;
          count++;
        }
      }
      BLADNN_VLOG(1) << "Tuned kernel impl " << count
                     << " kernel load from local lib: " << local_lib_path;
      closedir(dir);
      initialized_ = true;
      pthread_rwlock_unlock(&lock_);
      return true;
    }
  }
  BLADNN_LOG(WARNING) << "Tuned kernels lib path not found.";
  initialized_ = true;
  pthread_rwlock_unlock(&lock_);
  return false;
}

TVMWorkSpace::TVMWorkSpace() {
  ROCM_CALL(hipMalloc(reinterpret_cast<void**>(&buffer_), capacity_));
};

TVMWorkSpace::~TVMWorkSpace() { ROCM_CALL(hipFree(buffer_)); }

std::vector<void*> TVMWorkSpace::Allocate(
    const std::vector<TVMStorageInfo>& sizes) {
  mtx_.lock();
  std::vector<void*> ptrs;
  for (auto& info : sizes) {
    auto size = info.size;
    size_t aligned = ((size - 1) + alignment_) & (~(alignment_ - 1));
    if (capacity_ - tail_ < aligned) {
      return std::vector<void*>();
    }
    tail_ += aligned;
    used_ += aligned;
    ptrs.emplace_back(reinterpret_cast<char*>(buffer_) + tail_ - aligned);
  }
  return std::move(ptrs);
}

// Free should be used in pair with Allocate
void TVMWorkSpace::Free() {
  tail_ = 0;
  used_ = 0;
  mtx_.unlock();
}

LaunchParamConfig::LaunchParamConfig(
    size_t base, const std::vector<std::string>& launch_param_tags) {
  base_ = base;
  // Totally 6 parameters to be filled threadIdx.x(y,z) and blockIdx.x(y,z).
  std::vector<bool> filled(6, false);
  for (size_t i = 0; i < launch_param_tags.size(); ++i) {
    const std::string& tag = launch_param_tags[i];
    if (tag == kUseDynamicSharedMemoryTag) {
      BLADNN_CHECK(i == launch_param_tags.size() - 1)
          << "kUseDynamicSharedMemoryTag should be the last tag in "
             "launch_param_tags.";
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

ThreadWorkLoad LaunchParamConfig::Extract(
    const std::vector<uint64_t>& args) const {
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

}  // namespace tvm
}  // namespace bladnn
