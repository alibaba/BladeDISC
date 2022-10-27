# BladeNN_TVM交互python包disc_opt的编译
```bash
python build.py --rocm $ROCM_ROOT_PATH --tao_bridge  $TAO_BRIDGE_PATH   --tao_compiler $TAO_COMPILER_PATH
```
之后在dist中会产生编译好的disc_opt的wheel包

# BladeNN_TVM交互python包disc_opt的使用
参见example/network.py中disc_opt.DiscHandler的相关部分

# BladeNN_TVM功能介绍
BladeNN目前支持在AMD or DCU环境下开启TVM kernel优化支持，目前支持Gemm算子，调用方式为
```cpp
bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, void* c_ptr,
          int c_dim0, int c_dim1, int batch_count = 1, bool a_is_const = false,
          bool b_is_const = false);
```

# BladeNN_TVM接入方式
在BladeNN的TVM backend中，首先一个全局TVMHandler会针对不同的Device + Op的组合分别例化不同的kernel cache，cache中存放进行不同尺寸Op kernel计算所需的的hip function。用户调用gemm接口时，会根据所调用的gemm尺寸在cache中找到对应的hip function进行计算，如果function不存在或者调用失败，则gemm接口返回false，使用者可以fallback回基本的blas库进行Op的执行。
kernel cache在例化时，会在环境变量`BLADNN_KERNEL_CACHE`指向的路径中获取各个TVM Op kernel的binary与meta data文件，加载到cache中。每个TVM Op kernel对应一个`.hsaco`的hip function binary和一个`.meta_for_tao.json`的meta data info信息，例如：

- dcu_gemm_25_50_11776_1_0_0_float64_float64_float64_float64.hsaco
   - TVM编译生成的固定尺寸的Op kernel的binary，可以直接由hip runtime进行加载调用
   - 目前的命名方式为`{device}_{gemm}_{m}_{n}_{k}_{transpose_a}_{transpose_b}_{transpose_c}_{dtype_a}_{dtype_b}_{dtype_c}_{dtype_parameter}`的格式
- dcu_gemm_25_50_11776_1_0_0_float64_float64_float64_float64.meta_for_tao.json
   - hsaco中function调用时需要的参数信息，根据其中提供的参数可以正确地调用TVM生成的Op kernel
   - meta_for_tao.json的格式如下：
      - `func_info`中存放对应的hsaco中需要用到的function的名称，以`default_function_kernel{id}`命名，执行Op kernel时，按照id的次序依次调用各个function从而完成整个Op的执行；
      - `launch_param_tags`和`launch_param_args`一一对应，表示该function launch时候需要给出的block和thread维度的launch dimension size；
      - `arg_types`表示该function调用时给出的arguments，通常为函数执行的输入输出data指针，`handle`表示指针类型；
      - `storage`表示调用各个function完成整个Op执行时所需的额外workspace buffer的显存空间，描述了worksapce buffer所需的data type和data elements的个数，总所需空间为sizeof(datatype) * data elements个数；
      - 此部分meta data json的内容命名与定义参考了TVM，基本与TVM中保持一致。
```json
{
  "tvm_version": "0.1.0", 
  "func_info": {
    "default_function_kernel0": {
      "name": "", 
      "arg_types": ["handle", "handle", "handle"], 
      "launch_param_tags": ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.x"], 
      "launch_param_args": [32, 25, 4, 25, 2, 25]
    }, 
    "default_function_kernel1": {
      "name": "", 
      "arg_types": ["handle", "handle"], 
      "launch_param_tags": ["blockIdx.y", "blockIdx.x", "threadIdx.x", "threadIdx.y"], 
      "launch_param_args": [50, 1, 100, 8]
    }
  }, 
  "storage": {
    "T_matmul_TN.rf": ["float64", "160000"]
  }
}
```
BladeNN通过从环境变量`BLADNN_KERNEL_CACHE`指向的路径中读取不同kernel的`.hsaco`和`.meta_for_tao.json`文件，获取各个Op kernel的执行function与调用方式，在运行时即可调用TVM生成的kernel进行Op的执行。

# BladeNN_TVM Tuning方式
Step1：
进行tuning时，首先会判断环境变量`BLADNN_COLLECT_STATUS`的值，如果为`1`，那么BladeNN的gemm调用时会收集各个Op调用的具体信息。
Step2：
之后当`BLADNN_COLLECT_STATUS`的值为`2`时，BladeNN会将所收集的Op调用信息dump到环境变量`BLADNN_COLLECT_CACHE`所指向的路径中的`kernel_info.txt`的文件中，Op的dump格式与TVM kernel加载文件的命名方式保持一致，即`{device}_{gemm}_{m}_{n}_{k}_{transpose_a}_{transpose_b}_{transpose_c}_{dtype_a}_{dtype_b}_{dtype_c}_{dtype_parameter}`。此时，disc_opt python包中的Tuning Flow会针对`kernel_info.txt`文件中所描述的Op kernel进行TVM部分的Codegen且与rocblas的baseline是先进行比较，将有优势的TVM kernel实现的`.hsaco`和`.meta_for_tao.json`文件存放在`BLADNN_KERNEL_CACHE`指向的路径中。
Step3：
结束Tuning Flow后，`BLADNN_COLLECT_STATUS`会从`2`置为`0`，当BladeNN判断`BLADNN_COLLECT_STATUS`从`2`变为其他值时，表示Tuning Flow的结束和TVM kernel文件的更新，此时BladeNN会重新读取`BLADNN_KERNEL_CACHE`中的文件更新cache，进行cache的ReInit。之后当`BLADNN_COLLECT_STATUS`的值为`0`时，BladeNN会进行正常的Kernel Op执行，不进行Op信息收集和tuning的操作，使用Tuning Flow生成的TVM kernel实现进行执行。

- 此部分的Step控制均由disc_opt python包中的Tuning Flow进行控制

# BladeNN_TVM环境变量

- BLADNN_KERNEL_CACHE

存放TVM Op kernel实现相应的`.hsaco`和`.meta_for_tao.json`的路径

- BLADNN_COLLECT_STATUS

表示Tuning时的step状态：

      - 0：正常执行
      - 1：收集Op信息
      - 2：dump 收集的Op信息并完成Tuning Flow
- BLADNN_COLLECT_CACHE

所收集的Op信息所dump至的路径，为`{BLADNN_COLLECT_CACHE}/kernel_info.txt`

- BLADNN_DEVICE_TYPE

指定Device类型，如果不指定，BladeNN会自动检测GPU的型号