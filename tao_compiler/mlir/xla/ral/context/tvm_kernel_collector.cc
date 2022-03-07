
#include "tensorflow/compiler/mlir/xla/ral/context/tvm_kernel_collector.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_logging.h"
#include <iostream>
#include <fstream>
#include "tensorflow/compiler/mlir/xla/ral/context/tvm_kernel_cache.h"

#if TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"
#endif

#if TENSORFLOW_USE_ROCM
#define CUDA_SUCCESS hipSuccess
#endif

#define ROCM_CALL(func)                                              \
  {                                                                  \
    hipError_t e = (func);                                           \
    CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }



namespace tao {
namespace ral {
namespace tvm_impl  {

static KernelCollector kernel_collector_ins;

static bool GetCollectorEnable() {
   bool enable = false;
   const char* pro_str = getenv("DISC_KERNEL_PROFILING");
   if (pro_str) {
       enable = std::string(pro_str) == "1";
   } 
   return enable;
}

static std::string GetCacheLocation() {
    std::string path = ".";
    const char* pro_str = getenv("DISC_PROFILING_CACHE");
    if (pro_str) {
       path = std::string(pro_str);
    }     
    return path;
}

void CollectorCheckEnable() {
    kernel_collector_ins.CheckEnable();
}

void CollectorCheckDevice() {
    kernel_collector_ins.CheckDevice();
}

const std::string& CollectorDeviceStr() {
    // VLOG(0) << "Get device" <<  kernel_collector_ins.Device();
    return  kernel_collector_ins.Device();
}



template<typename InT, typename OutT, typename AlphaBeta> 
void CollectorAddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b) {  
    kernel_collector_ins.AddGemmKernel<InT, OutT, AlphaBeta>(device, m, n, k, trans_a, trans_b);
}

void CollectorDumpResults() {
    kernel_collector_ins.DumpResults();
}

void CollectorAddKernel(const std::string& kernel_key) {
    kernel_collector_ins.AddKernel(kernel_key);
}

KernelCollector::KernelCollector() {
   CheckEnable();
}

void KernelCollector::CheckEnable() {
    std::lock_guard<std::mutex> lck(mtx_);
    auto pre_enable = enable_;
    enable_ = GetCollectorEnable();
    if (pre_enable && !enable_) {
        // VLOG(0) << "clear #####################";
        kernels_.clear();
    }
    if (pre_enable && !enable_) {
        // VLOG(0) << "reinit #####################";
        tvm_impl::TVMFuncCacheReInit();
    }
    TAO_VLOG(1) << "Collector check enable " << (enable_?1:0);
}

void KernelCollector::DumpResults() {
    std::lock_guard<std::mutex> lck(mtx_);
    if (!enable_) {
        return;
    }
    auto path = GetCacheLocation() + "/kernel_info.txt";
    std::ofstream file(path);
    std::ostringstream str;
    // VLOG(0) << "kernel size " << kernels_.size();
    for (auto it : kernels_) {
        str << it << std::endl;
    }
    // TAO_VLOG(0) << "Collected Kernel:\n" << str.str();
    file << str.str();
    file.close();
    TAO_VLOG(1) << "Collected Kernel to " << path << " with size " << kernels_.size();
}

KernelCollector::~KernelCollector() {
    DumpResults();
}

template<typename InT, typename OutT, typename AlphaBeta>
void KernelCollector::AddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b) {
     if (enable_) {
        auto ta = trans_a ? tvm_impl::TVMGemmTranspose::Transpose
                    : tvm_impl::TVMGemmTranspose::NoTranspose;
        auto tb = trans_b ? tvm_impl::TVMGemmTranspose::Transpose
                                            : tvm_impl::TVMGemmTranspose::NoTranspose;
        auto key = tvm_impl::GetGemmTVMFuncKey<InT, OutT, AlphaBeta>(device, m, n, k, ta, tb);
        AddKernel(key);
     }
    //  VLOG(0) << "kernel name " << (uint64_t)this << " " << kernels_.size();
}


void KernelCollector::AddKernel(const std::string& kernel_key) {
    if (enable_) { 
        std::lock_guard<std::mutex> lck(mtx_);
        kernels_.emplace(kernel_key);
    }
}

void KernelCollector::CheckDevice() {
    std::lock_guard<std::mutex> lck(mtx_);
    const char* pro_str = getenv("DISC_PROFILING_DEVICE");
    if (pro_str) {
       device_env_ = std::string(pro_str);
    } 
    if (device_env_.empty()) {
#if TENSORFLOW_USE_ROCM
        int device_id;
        ROCM_CALL(hipGetDevice(&device_id));
        hipDeviceProp_t prop;
        ROCM_CALL(hipGetDeviceProperties(&prop, device_id));
        auto arch_type = prop.gcnArch;
        device_env_ = "gfx" + std::to_string(arch_type); //906";
        if(device_env_ == "gfx910") {
            device_env_ = "gfx90a";
        }
#endif
    }
};

const std::string& KernelCollector::Device() {
    return device_env_;
};


template
void KernelCollector::AddGemmKernel<float, float, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void KernelCollector::AddGemmKernel<double, double, double>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void KernelCollector::AddGemmKernel<Eigen::half, Eigen::half, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<float, float, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<double, double, double>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<Eigen::half, Eigen::half, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

}
}
}