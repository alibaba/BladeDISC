#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

//#define MOD
#define KNAME "_Z6kernelPdS_mmmmmmmmmS_S_mmm"
#define KNAME1 "main_kColReduction_reduce__2_1_0___8w32h_1"


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
#define D0 11776
#define D1 25
#define D2 50
#define L0 256
#define L1 1472

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
    std::string name = KNAME1;
    hipError_t result = hipModuleGetFunction(&func, module, name.c_str());
    if (result != hipSuccess) {
      std::cout << "ROCMError: hipModuleGetFunction " << name
                 << " failed with error: " << hipGetErrorString(result) << std::endl;
      name = KNAME;
      result = hipModuleGetFunction(&func, module, name.c_str());
    if (result != hipSuccess) 
      std::cout << "ROCMError: hipModuleGetFunction " << name
                 << " failed with error: " << hipGetErrorString(result) << std::endl;
    }
    return func;
}
 
void make_buffer(void* ptr, std::vector<int> dims, std::vector<uint64_t>& datas) {
    datas.push_back(reinterpret_cast<uint64_t>(ptr)); 
    datas.push_back(reinterpret_cast<uint64_t>(ptr)); 
    datas.push_back(0);
    int pre_size = datas.size(); 
    datas.resize(pre_size + dims.size() * 2);
    int stride = 1;
    for (int i  = 0; i < dims.size(); i++) {
      datas[pre_size + i] = dims[i];
      datas[pre_size + 2 * dims.size() - 1 - i] = stride;
      stride *=  dims[dims.size() - 1 - i];
    }
} 
// arg0 = 25
// arg1 = [11776, 25]
// arg2 = 256
// arg3 = 376832
// arg4 = 4
// arg5 = [11776, 25]
// arg6 = 11776*25
// arg7 = 0
// arg8 = 0
// arg9 = 1
// arg10 = 1
// arg11 = 50
// arg12 = 11776
// arg13 = [11776, 50]
// arg14 = 0
// arg15 = 25
// arg16 = [11776, 25]
// arg17 = [11776, 25]
// arg18 = [11776, 25]
// arg19 = 11776
// arg20 = 25
// arg21 = [11776, 25]
// arg22 = [25]



std::vector<void*> make_args(double ** ptr, double * input, double * output, double * output_reduce) {
  std::vector<uint64_t>* datas_ptr = new std::vector<uint64_t>();
  auto& datas = *datas_ptr;
  // datas.push_back(D1);
  std::vector<int> shape0 = {D0, D1};
  std::vector<int> shape1 = {D0, D2};
  make_buffer((void*)ptr[0], shape0, datas);
  std::vector<uint64_t> tmp = {L0, L1 * L0, 4, D0 * D1};
  datas.insert(datas.end(), tmp.begin(), tmp.end());
  // make_buffer((void*)ptr[1], shape0, datas);
  // tmp = {D0*D1, 0, 0, 1, 1, D2, D0};
  // datas.insert(datas.end(), tmp.begin(), tmp.end());
  // make_buffer((void*)input, shape1, datas);
  // tmp = {0, D1};
  // datas.insert(datas.end(), tmp.begin(), tmp.end());
  // make_buffer((void*)ptr[2], shape0, datas);
  // make_buffer((void*)ptr[3], shape0, datas);
  // make_buffer((void*)ptr[4], shape0, datas);
  // tmp = {D0, D1};
  // datas.insert(datas.end(), tmp.begin(), tmp.end());
  // make_buffer((void*)output, shape0, datas);
  make_buffer((void*)output_reduce, {D1}, datas);
  std::vector<void*> args;
  for (int i = 0; i < datas.size(); i++) {
    // std::cout << datas[i] << std::endl;
	  args.push_back(static_cast<void*>(static_cast<uint64_t *>(datas.data()) + i));
    // std::cout << args.back() << std::endl;
  }
  int idx = 0;
  // for (auto &i : args) {
  //   std::cout << datas[idx] << " : " << i  << " : " << reinterpret_cast<uint64_t>(*static_cast<const uint64_t*>(i)) << std::endl;
  //   idx++;
  // }
  return std::move(args);
}



void LaunchKernel(hipFunction_t func, void ** arg_addr) {

#ifdef MOD
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

    ROCM_DRIVER_CALL(hipModuleLaunchKernel(func, L1, 1, 1, 
                L0, 1, 1, 
                0,
                0, 
		            arg_addr, nullptr
		));
}




int main(int argc, char* argv[]) {
  const int d0 = D0;
  const int d1 = D1;
  const int d2 = D2;
  double* ptr[5];
  double* input; 
  double* output, *output0;
  uint64_t mem_size = d0 * d1 * sizeof(double);
  uint64_t mem_size0 = d0 * d2 * sizeof(double);
  uint64_t mem_size1 = d2 * sizeof(double);


  for (int i = 0; i < 5; i++) {
    checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&ptr[i]), mem_size));
  }
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&input), mem_size0));

  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output0), mem_size1));
  
  double* h_init = reinterpret_cast<double*>(malloc(mem_size0));
  for (int64_t i = 0; i < mem_size0/sizeof(double); i++) {
    h_init[i] = 0.001;
  }
  for (int i = 0; i < 5; i++) {
    checkCudaErrors(hipMemcpy(ptr[i], h_init, mem_size, hipMemcpyDefault));
  }
  checkCudaErrors(hipMemcpy(input, h_init, mem_size0, hipMemcpyDefault));
  checkCudaErrors(hipMemset(output, 0, mem_size));
  checkCudaErrors(hipMemset(output0, 0, mem_size1));

  // free(h_init);
//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  hipEvent_t start, stop;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));
  std::vector<void*> args = make_args(ptr, input, output, output0);
  int idx = 0;
  // for (auto &i : args) {
  //   std::cout << (*datas)[idx] << " " << i  << " : " << reinterpret_cast<uint64_t>(*static_cast<const uint64_t*>(i)) << std::endl;
  //   idx++;
  // }

  hipFunction_t func = GetFunc(argv[1]);
  double* o_red = reinterpret_cast<double*>(malloc(mem_size1));
  memset(o_red, 0, mem_size1);
  double* o_init = reinterpret_cast<double*>(malloc(mem_size));
  memset(o_init, 0, mem_size);

  // warmup
  checkCudaErrors(hipEventRecord(start));
  int times = 100;


  for (int i = 0; i < times; i++) {
    // for (int64_t j = 0; j < mem_size/sizeof(float); j++) {
    //   h_init[j] = 0.001  *( i);
    // }
    // checkCudaErrors(hipMemcpyHtoD(input, h_init, mem_size));
    // copy<<<d0*d1/launch_dim, launch_dim>>>(input, inputt);
    // softmax1st<<<1, launch_dim>>>(input, output);
    LaunchKernel(func, reinterpret_cast<void **>(args.data()) );
    // copy<<<1, d0>>>(outputt, output);
    // checkCudaErrors(hipMemcpyDtoH(o_init, output, mem_size0));
  // for (int64_t j = 0; j < d0; j++) {
  //     printf(" %f ", o_init[j]);
  // }
  }
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec / times);

  checkCudaErrors(hipMemcpy(o_init, output, mem_size, hipMemcpyDefault));
  checkCudaErrors(hipMemcpy(o_red, output0, mem_size1, hipMemcpyDefault));
    // checkCudaErrors(hipMemcpyDtoH(o_mid, mid, mem_size));
  // printf("\n");
  // for (int64_t i = 0; i < 100; i++) {
	//  printf(" %f ", o_init[i]);
  // }
  printf("\n");
  for (int64_t i = 0; i < 25; i++) {
	 printf(" %f ", o_red[i]);
  }
  printf("\n");


  return 0;
}
