

#include "bladnn/backend/tvm/tvm_collector.h"

#include <fstream>
#include <iostream>

#include "bladnn/backend/tvm/tvm_handle.h"
#include "bladnn/utils/log.h"
#include "hip/hip_runtime.h"

#define CUDA_SUCCESS hipSuccess

#define ROCM_CALL(func)                                                    \
  {                                                                        \
    hipError_t e = (func);                                                 \
    BLADNN_CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }

namespace bladnn {
namespace tvm {

static std::string GetCacheLocation() {
  std::string path = ".";
  const char* pro_str = getenv("BLADNN_COLLECT_CACHE");
  if (pro_str != nullptr) {
    path = std::string(pro_str);
  }
  return path;
}

void KernelCollector::DumpResults() {
  std::lock_guard<std::mutex> lck(mtx_);
  auto path = GetCacheLocation() + "/kernel_info.txt";
  std::ofstream file(path);
  std::ostringstream str;
  // VBLADNN_LOG(0) << "kernel size " << kernels_.size();
  for (auto it : kernels_) {
    str << it << std::endl;
  }
  // TAO_VBLADNN_LOG(0) << "Collected Kernel:\n" << str.str();
  file << str.str();
  file.close();
  BLADNN_VLOG(0) << "Collected Kernel to " << path << " with size "
                 << kernels_.size();
}

KernelCollector::KernelCollector() {}

KernelCollector::~KernelCollector() {}

void KernelCollector::Clear() { kernels_.clear(); }

template <Dtype InT, Dtype OutT, Dtype AlphaBeta>
void KernelCollector::AddGemmKernel(const std::string& device, int64_t m,
                                    int64_t n, int64_t k, bool trans_a,
                                    bool trans_b) {
  auto ta =
      trans_a ? TVMGemmTranspose::Transpose : TVMGemmTranspose::NoTranspose;
  auto tb =
      trans_b ? TVMGemmTranspose::Transpose : TVMGemmTranspose::NoTranspose;
  auto key = TVMHandler::GetGemmTVMFuncKey<InT, OutT, AlphaBeta>(device, m, n,
                                                                 k, ta, tb);
  AddKernel(key);
}

void KernelCollector::AddKernel(const std::string& kernel_key) {
  std::lock_guard<std::mutex> lck(mtx_);
  kernels_.emplace(kernel_key);
}

template void
KernelCollector::AddGemmKernel<Dtype::kF32, Dtype::kF32, Dtype::kF32>(
    const std::string& device, int64_t m, int64_t n, int64_t k, bool trans_a,
    bool trans_b);

template void
KernelCollector::AddGemmKernel<Dtype::kF64, Dtype::kF64, Dtype::kF64>(
    const std::string& device, int64_t m, int64_t n, int64_t k, bool trans_a,
    bool trans_b);

}  // namespace tvm
}  // namespace bladnn
