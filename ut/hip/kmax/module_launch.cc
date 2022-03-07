#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

//#define MOD

#define ROCM_DRIVER_CALL(x)                                                                    \
  {                                                                                            \
    hipError_t result = x;                                                                     \
    if (result != hipSuccess && result != hipErrorDeinitialized) {                             \
      std::cout << "ROCM HIP Error: " #x " failed with error: " << hipGetErrorString(result) << std::endl; \
    }                                                                                          \
  }

#define ROCM_CALL(func)                                              \
  {                                                                  \
    hipError_t e = (func);                                           \
    CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }
#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)
#define D0 1
#define D1 4096
#define warp_size 32
#define launch_dim (8*warp_size)

__global__ void copy(float* input, float* output) {
   output[blockIdx.x * blockDim.x + threadIdx.x]  = input[blockIdx.x * blockDim.x + threadIdx.x] ;
}

void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

hipFunction_t GetFunc(const std::string& file_name)  {
    std::string data;
    hipModule_t module;
    LoadBinaryFromFile(file_name, &data);
    ROCM_DRIVER_CALL(hipModuleLoadData(&module, data.c_str()));
    hipFunction_t func;
    std::string name = "_Z10softmax1stPfS_";
    name = "_Z10softmax1stPfS_i";
    name = "_Z10softmax1stPfS_iiS_";
//    name = "_Z25__device_stub__softmax1stPfS_iiS_";
    name = "main_kRowReduction_reduce__2_1_0___1b1rX_vectile2X_no_vectile";
    hipError_t result = hipModuleGetFunction(&func, module, name.c_str());
    if (result != hipSuccess) {
      std::cout << "ROCMError: hipModuleGetFunction " << name
                 << " failed with error: " << hipGetErrorString(result) << std::endl;
      name = "_Z61main_kRowReduction_reduce__2_1_0___1b1rX_vectile2X_no_vectilePfS_iiiiiiiS_S_iii";
      result = hipModuleGetFunction(&func, module, name.c_str());
    if (result != hipSuccess) 
      std::cout << "ROCMError: hipModuleGetFunction " << name
                 << " failed with error: " << hipGetErrorString(result) << std::endl;
    }
    return func;
}

void LaunchKernel(hipFunction_t func, int d0, int d1, float* input, float* output, float* mid) {
    int arg0 = launch_dim;
    int arg1 = d0 * launch_dim;

#ifdef MOD
    uint8_t args[32];
    size_t size = 32;
  //  void *args[] = {&input, &arg0, &arg1, &output};
    *((float**)(&args[0])) = input;
    *((float**)(&args[8])) = output;
    *((int*)(&args[16])) = launch_dim;
    *((int*)(&args[20])) = D0 * launch_dim;
    *((int*)(&args[24])) = D1;
#else
    uint8_t args[112];
    size_t size = 112;
  //  void *args[] = {&input, &arg0, &arg1, &output};
    
    *((float**)(&args[0])) = input;
    *((float**)(&args[8])) = input;
    *((uint64_t*)(&args[16])) = 0;
    *((uint64_t*)(&args[24])) = D0;
    *((uint64_t*)(&args[32])) = D1;
    *((uint64_t*)(&args[40])) = D1;
    *((uint64_t*)(&args[48])) = 1;

    *((uint64_t*)(&args[56])) = launch_dim;
    *((uint64_t*)(&args[64])) = D0 * launch_dim;
    *((float**)(&args[72])) = output;
    *((float**)(&args[80])) = output;
    *((uint64_t*)(&args[88])) = 0;
    *((uint64_t*)(&args[96])) = D0;
    *((uint64_t*)(&args[104])) = 1;

    std::vector<void*> arg_addr;
    for (int i = 0; i < 14; i++) {
	arg_addr.push_back(static_cast<void*>(&args[8*i]));
    }


#endif
/*    std::cout <<  
	   (*(float**)(&args[0])) <<  " " <<
(*(int*)(&args[8])) << " " <<
(*(int*)(&args[12])) << " " <<
	   (*(float**)(&args[16])) <<  " " << std::endl;*/
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*)args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &size, HIP_LAUNCH_PARAM_END};

    ROCM_DRIVER_CALL(hipModuleLaunchKernel(func, d0, 1, 1, 
                launch_dim, 1, 1, 
                0,
                0, 
//		nullptr, reinterpret_cast<void**>(&config)
		reinterpret_cast<void**>(arg_addr.data()), nullptr
		));
		//nullptr));

   // ROCM_DRIVER_CALL(hipLaunchKernel(func, dim3(d0), dim3(launch_dim), args, 0, NULL));
}




int main(int argc, char* argv[]) {
  const int d0 = D0;
  const int d1 = D1;

  float *input, *output, *mid;
  float *inputt, *outputt;
  uint64_t mem_size = d0 * d1 * sizeof(float);
  uint64_t mem_size0 = d0 * sizeof(float);
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&input), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size0));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&inputt), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&outputt), mem_size0));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&mid), mem_size));
  float* h_init = reinterpret_cast<float*>(malloc(mem_size));
  for (int64_t i = 0; i < mem_size/sizeof(float); i++) {
    h_init[i] = 0.001*i;
  }
  checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));
  checkCudaErrors(hipMemset(output, 0, mem_size0));
  checkCudaErrors(hipMemset(mid, 0, mem_size));

  free(h_init);
//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  hipEvent_t start, stop;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));
  hipFunction_t func = GetFunc(argv[1]);
  float* o_mid = reinterpret_cast<float*>(malloc(mem_size));
  memset(o_mid, 0, mem_size);
  float* o_init = reinterpret_cast<float*>(malloc(mem_size0));
  memset(o_init, 0, mem_size0);

  // warmup
  checkCudaErrors(hipEventRecord(start));
  int times = 100;
  for (int i = 0; i < times; i++) {
    for (int64_t j = 0; j < mem_size/sizeof(float); j++) {
      h_init[j] = 0.001  *( i);
    }
    checkCudaErrors(hipMemcpyHtoD(input, h_init, mem_size));
    copy<<<d0*d1/launch_dim, launch_dim>>>(input, inputt);
    // softmax1st<<<1, launch_dim>>>(input, output);
    LaunchKernel(func, d0, d1, inputt, outputt, mid);
    copy<<<1, d0>>>(outputt, output);
    checkCudaErrors(hipMemcpyDtoH(o_init, output, mem_size0));
  for (int64_t j = 0; j < d0; j++) {
      printf(" %f ", o_init[j]);
  }
  }
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec);

    checkCudaErrors(hipMemcpyDtoH(o_mid, mid, mem_size));
  printf("\n");
  /*for (int64_t i = 0; i < d0*d1; i++) {
	 printf(" %f ", o_mid[i]);
  }
  printf("\n");*/


  return 0;
}
