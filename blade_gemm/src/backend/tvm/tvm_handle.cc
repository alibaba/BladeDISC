#include "bladnn/backend/tvm/tvm_handle.h"

#include <dirent.h>

#include <fstream>
#include <iostream>
#include <unordered_map>

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

static std::mutex mtx;
static TVMHandler* tvm_handler_ = nullptr;

TVMHandler* TVMHandlerCreateOrGet() {
  std::lock_guard<std::mutex> lck(mtx);
  if (tvm_handler_ == nullptr) {
    tvm_handler_ = new TVMHandler();
  }
  return tvm_handler_;
}

static void GetCollectorEnable(bool* collect, bool* dump) {
  *collect = false;
  *dump = false;
  const char* pro_str = getenv("BLADNN_COLLECT_STATUS");
  if (pro_str != nullptr) {
    *collect = pro_str[0] == '1';
    *dump = pro_str[0] == '2';
  }
}

template <Dtype DType>
inline std::string GetTVMFuncKey() {
  BLADNN_LOG(FATAL) << "Not supported OPT func key type.";
  return "";
}

template <>
inline std::string GetTVMFuncKey<Dtype::kF32>() {
  return "float32";
}

template <>
inline std::string GetTVMFuncKey<Dtype::kF64>() {
  return "float64";
}

template <Dtype a_type, Dtype b_type, Dtype c_type>
inline std::string TVMHandler::GetGemmTVMFuncKey(const std::string& device,
                                                 int64_t m, int64_t n,
                                                 int64_t k,
                                                 TVMGemmTranspose trans_a,
                                                 TVMGemmTranspose trans_b) {
  char buffer[1024];
  sprintf(buffer, "%s_gemm_%d_%d_%d_%d_%d_%d_%s_%s_%s_%s", device.c_str(), m, n,
          k, trans_a, trans_b, TVMGemmTranspose::NoTranspose,
          GetTVMFuncKey<a_type>().c_str(), GetTVMFuncKey<b_type>().c_str(),
          GetTVMFuncKey<c_type>().c_str(), GetTVMFuncKey<c_type>().c_str());
  auto ss = std::string(buffer);
  return ss;
}

TVMHandler::TVMHandler() {
  CheckGPUDevice();
  CheckStatus();
}

TVMFuncCache* TVMHandler::TVMFuncCacheCreateOrGet(const std::string& op,
                                                  const std::string& device) {
  return tvm_func_caches_.CreateOrGet(op, device);
}
void TVMHandler::TVMFuncCacheReInit() { tvm_func_caches_.ReInit(); }

void TVMHandler::CheckStatus() {
  std::lock_guard<std::mutex> lck(mtx_);
  auto enable = false;
  bool dump = false;
  GetCollectorEnable(&enable, &dump);
  if (!dump_ && dump) {
    kernel_collector_ins_.DumpResults();
    kernel_collector_ins_.Clear();
  }
  if (dump_ && !dump) {
    BLADNN_VLOG(0) << "Reinit fucn cache.";
    TVMFuncCacheReInit();
  }
  enable_ = enable;
  dump_ = dump;
  BLADNN_VLOG(1) << "Collector check enable " << (enable_ ? 1 : 0) << " dump "
                 << (dump_ ? 1 : 0);
}

void TVMHandler::CheckGPUDevice() {
  std::lock_guard<std::mutex> lck(mtx_);
  const char* pro_str = getenv("BLADNN_DEVICE_TYPE");
  if (pro_str != nullptr) {
    gpu_device_ = std::string(pro_str);
  }
  if (gpu_device_.empty()) {
    int device_id;
    ROCM_CALL(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    ROCM_CALL(hipGetDeviceProperties(&prop, device_id));
    auto arch_type = prop.gcnArch;
    gpu_device_ = "gfx" + std::to_string(arch_type);  // 906";
    if (gpu_device_ == "gfx910") {
      gpu_device_ = "gfx90a";
    }
  }
};

const std::string& TVMHandler::GPUDevice() { return gpu_device_; };

template <Dtype InT, Dtype OutT, Dtype AlphaBeta>
void TVMHandler::CollectorAddGemmKernel(const std::string& device, int64_t m,
                                        int64_t n, int64_t k, bool trans_a,
                                        bool trans_b) {
  if (enable_) {
    kernel_collector_ins_.AddGemmKernel<InT, OutT, AlphaBeta>(device, m, n, k,
                                                              trans_a, trans_b);
  }
}

template void
TVMHandler::CollectorAddGemmKernel<Dtype::kF32, Dtype::kF32, Dtype::kF32>(
    const std::string& device, int64_t m, int64_t n, int64_t k, bool trans_a,
    bool trans_b);

template void
TVMHandler::CollectorAddGemmKernel<Dtype::kF64, Dtype::kF64, Dtype::kF64>(
    const std::string& device, int64_t m, int64_t n, int64_t k, bool trans_a,
    bool trans_b);

template std::string
TVMHandler::GetGemmTVMFuncKey<Dtype::kF32, Dtype::kF32, Dtype::kF32>(
    const std::string& device, int64_t m, int64_t n, int64_t k,
    TVMGemmTranspose trans_a, TVMGemmTranspose trans_b);

template std::string
TVMHandler::GetGemmTVMFuncKey<Dtype::kF64, Dtype::kF64, Dtype::kF64>(
    const std::string& device, int64_t m, int64_t n, int64_t k,
    TVMGemmTranspose trans_a, TVMGemmTranspose trans_b);

void TVMHandler::CollectorDumpResults() { kernel_collector_ins_.DumpResults(); }

void TVMHandler::CollectorAddKernel(const std::string& kernel_key) {
  if (enable_) {
    kernel_collector_ins_.AddKernel(kernel_key);
  }
}

}  // namespace tvm
}  // namespace bladnn
