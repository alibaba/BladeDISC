#pragma once

#include <mutex>
#include <set>
#include <string>

#include "bladnn/bladnn.h"

namespace bladnn {
namespace tvm {

class KernelCollector {
 public:
  KernelCollector();
  ~KernelCollector();
  void AddKernel(const std::string& kernel_key);
  template <Dtype InT, Dtype OutT, Dtype AlphaBeta>
  void AddGemmKernel(const std::string& device, int64_t m, int64_t n, int64_t k,
                     bool trans_a, bool trans_b);
  void DumpResults();
  void Clear();

 private:
  std::mutex mtx_;
  std::set<std::string> kernels_;
};

}  // namespace tvm
}  // namespace bladnn
