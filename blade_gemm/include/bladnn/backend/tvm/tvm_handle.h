#pragma once

#include <bladnn/bladnn.h>
#include <bladnn/utils/log.h>
#include <hip/hip_runtime.h>
#include <pthread.h>

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "bladnn/backend/tvm/tvm_cache.h"
#include "bladnn/backend/tvm/tvm_collector.h"

namespace bladnn {
namespace tvm {

class TVMHandler {
 public:
  TVMHandler();
  TVMFuncCache* TVMFuncCacheCreateOrGet(const std::string& op,
                                        const std::string& device);
  void TVMFuncCacheReInit();
  template <::bladnn::Dtype a_type, ::bladnn::Dtype b_type,
            ::bladnn::Dtype c_type>
  inline static std::string GetGemmTVMFuncKey(const std::string& device,
                                              int64_t m, int64_t n, int64_t k,
                                              TVMGemmTranspose trans_a,
                                              TVMGemmTranspose trans_b);
  template <Dtype InT, Dtype OutT, Dtype AlphaBeta>
  void CollectorAddGemmKernel(const std::string& device, int64_t m, int64_t n,
                              int64_t k, bool trans_a, bool trans_b);
  void CheckStatus();
  const std::string& GPUDevice();
  void CheckGPUDevice();
  void CollectorAddKernel(const std::string& kernel_key);
  void CollectorDumpResults();

 private:
  bool enable_ = false;
  bool dump_ = false;
  std::mutex mtx_;
  TVMFuncCaches tvm_func_caches_;
  KernelCollector kernel_collector_ins_;
  std::string gpu_device_;
};

TVMHandler* TVMHandlerCreateOrGet();

}  // namespace tvm
}  // namespace bladnn
