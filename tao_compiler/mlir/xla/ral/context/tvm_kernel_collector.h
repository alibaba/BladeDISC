#ifndef RAL_CONTEXT_TVM_KERNEL_COLLECTOR_H_
#define RAL_CONTEXT_TVM_KERNEL_COLLECTOR_H_


#include <set>
#include <string>
#include <mutex>

namespace tao {
namespace ral {
namespace tvm_impl {

void CollectorAddKernel(const std::string& kernel_key);
void CollectorCheckEnable();
template<typename InT, typename OutT, typename AlphaBeta> 
void CollectorAddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b); 
void CollectorDumpResults();
void CollectorCheckDevice();
const std::string& CollectorDeviceStr();


class KernelCollector {    
public:   
    KernelCollector();
    ~KernelCollector();
    void AddKernel(const std::string& kernel_key);
    void CheckEnable();
    void CheckDevice();
    const std::string& Device();
    
    template<typename InT, typename OutT, typename AlphaBeta> 
    void AddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);
    void DumpResults();

private:
    std::mutex mtx_;
    std::set<std::string> kernels_;
    bool enable_;
    std::string device_env_;
};

}
}
}

#endif