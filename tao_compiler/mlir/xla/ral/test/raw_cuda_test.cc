// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "cuda.h"
#include "mlir/xla/ral/context/base/cuda/cuda_context_impl.h"

using namespace tao::ral;

// A quick test for raw cuda ral context
// step 1:
//   bazel build //tensorflow/compiler/mlir:dhlo_compiler_main
// step 2:
//   TAO_ENABLE_RAL=true bazel-bin/tensorflow/compiler/mlir/dhlo_compiler_main
//   mlir/xla/raltest/test.mlir
// step 3:
//   bazel build //tensorflow/compiler/mlir/xla/ral:raw_cuda_test
// step 4:
//   bazel build //tensorflow/compiler/mlir/xla/ral:libral_base_context.so
// step 5: h/d for host/device
//   bazel-bin/mlir/xla/ralraw_cuda_test
//   bazel-bin/mlir/xla/rallibral_base_context.so ./out.so
//   -o1 -i2 2x3xf32_h 2x3xf32_d [f32_d]

// In case you may need to debug llvm generated code:
// (here tmp.ll refers to the generated llvm ir after the last pass during
// step 2.) llvm-as tmp.ll llc --debugger-tune=gdb --relocation-model=pic
// --filetype=obj tmp.bc gcc tmp.o -shared -o out.so gdb
// bazel-bin/mlir/xla/ralraw_cuda_test
//   r bazel-bin/mlir/xla/rallibral_base_context.so
//   ./out.so

static int32_t reportErrorIfAny(CUresult result, const char* where) {
  if (result != CUDA_SUCCESS) {
    std::ostringstream out;
    out << "CUDA failed with " << result << " in " << where << std::endl;
    std::cout << "[[ ERROR ]]: " << out.str() << std::endl;
  }
  return result;
}

enum class ElemType { kUnknown, kF32, kI1, kI32 };

buffer_shape_t parseShape(char* s, ElemType* dtype, bool* is_host) {
  buffer_shape_t shape;
  std::vector<std::string> splitted = absl::StrSplit(s, 'x');
  for (int i = 0; i < splitted.size() - 1; ++i) {
    shape.emplace_back(std::stoi(splitted[i]));
  }
  std::vector<std::string> dtype_device = absl::StrSplit(splitted.back(), '_');
  if (dtype_device.front() == "f32") {
    *dtype = ElemType::kF32;
  } else if (dtype_device.front() == "i32") {
    *dtype = ElemType::kI32;
  } else if (dtype_device.front() == "i1") {
    *dtype = ElemType::kI1;
  } else {
    std::cout << "Error: dtype other than f32 is to be implemented\n";
  }
  if (dtype_device.back() == "h") {
    *is_host = true;
  } else if (dtype_device.back() == "d") {
    *is_host = false;
  } else {
    std::cout << "Error: placement of input tensor not properly assigned\n";
    std::cout << " format is like: 2x3xf32_{h|d}\n";
  }
  return shape;
}

void print_output_shape(void* d_result, const buffer_shape_t& shape) {
  std::cout << "out buffer = " << d_result << std::endl;
  std::cout << "out shape:\n" << std::endl;
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << "\tdim #" << i << ": " << shape[i] << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "Usage:\n\t"
              << "dhlo_compiler_main [path to lib] [path to compiled so file]"
              << "-o[num_of_outputs] -i[num_of_inputs] 16x32xf32 16x32xf32"
              << std::endl;
  }

  void* library_handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (!library_handle) {
    std::cout << ("fail to open ral library " + std::string(argv[1]))
              << " with err: " << dlerror() << std::endl;
    return 1;
  }

  void* tao_ral_func_ptr = dlsym(library_handle, "tao_ral_call_impl");
  if (!tao_ral_func_ptr) {
    std::cout << ("fail to find tao_ral_call_impl") << std::endl;
    return 1;
  }

  void* func_handle = dlopen(argv[2], RTLD_NOW);
  if (!func_handle) {
    std::cout << ("fail to open compiled binary " + std::string(argv[2]))
              << " with err: " << dlerror() << std::endl;
    return 1;
  }

  void* entry_func_ptr = dlsym(func_handle, "_mlir_lowered_tao_main");
  if (!entry_func_ptr) {
    std::cout << ("fail to find _mlir_lowered_tao_main") << std::endl;
    return 1;
  }
  using func_t = void (*)(void**);
  func_t entry_func = (func_t)entry_func_ptr;

  if ((argv[3][0] != '-') || (argv[3][1] != 'o')) {
    std::cout << "unexpected num_of_outputs";
    return 1;
  }
  int num_outputs = 0;
  sscanf(&argv[3][2], "%d", &num_outputs);
  std::cout << "num_outputs: " << num_outputs << std::endl;

  if ((argv[4][0] != '-') || (argv[4][1] != 'i')) {
    std::cout << "unexpected num_of_inputs";
    return 1;
  }
  int num_inputs = 0;
  sscanf(&argv[4][2], "%d", &num_inputs);
  std::cout << "num_inputs: " << num_inputs << std::endl;

  if (argc < 5 + num_inputs) {
    std::cout << "missing input shapes in arguments\n";
    std::cout << "Usage:\n\t"
              << "raw_cuda_test [path to lib] [path to compiled so file]"
              << "-o[num_of_outputs] -i[num_of_inputs] 16x32xf32 16x32xf32"
              << std::endl;
    return 1;
  }

  std::vector<buffer_shape_t> input_shapes;
  std::vector<ElemType> input_elem_types;
  std::vector<bool> input_placement;
  for (int idx = 0; idx < num_inputs; ++idx) {
    ElemType dtype = ElemType::kUnknown;
    bool is_host = false;
    buffer_shape_t shape = parseShape(argv[5 + idx], &dtype, &is_host);
    std::cout << "parsed shape: ";
    for (auto dim : shape) {
      std::cout << dim << ",";
    }
    std::cout << std::endl;
    input_shapes.emplace_back(shape);
    input_elem_types.emplace_back(dtype);
    input_placement.emplace_back(is_host);
  }

  std::vector<ElemType> out_elem_types;
  std::vector<bool> output_placement;
  if (argc < 5 + num_inputs + num_outputs) {
    // default output element type if not assigned
    for (int idx = 0; idx < num_outputs; ++idx) {
      std::cout << "output element type is not assigned,"
                << " and f32 is used by default" << std::endl;
      out_elem_types.emplace_back(ElemType::kF32);
      output_placement.push_back(false);
    }
  } else {
    for (int idx = 0; idx < num_outputs; ++idx) {
      std::string o_type_str(argv[5 + num_inputs + idx]);
      std::vector<std::string> splitted = absl::StrSplit(o_type_str, '_');
      if (splitted.front() == "f32") {
        out_elem_types.emplace_back(ElemType::kF32);
      } else if (splitted.front() == "i32") {
        out_elem_types.emplace_back(ElemType::kI32);
      } else if (splitted.front() == "i1") {
        out_elem_types.emplace_back(ElemType::kI1);
      } else {
        std::cout << "unsupported output element type in raw_cuda_test"
                  << std::endl;
        return 1;
      }
      if (splitted.back() == "h") {
        output_placement.push_back(true);
      } else if (splitted.back() == "d") {
        output_placement.push_back(false);
      } else {
        std::cout
            << "Error: placement of output tensor not properly assigned\n";
        std::cout << " format is like: f32_{h|d}\n";
        return 1;
      }
    }
  }

  std::cout << "xxxxxxxxxxxxxxxx over ################" << std::endl;

  CUdevice device;
  CUcontext cu_ctx;
  reportErrorIfAny(cuInit(0), "cuInit");
  reportErrorIfAny(cuDeviceGet(&device, 0), "cuDeviceGet");
  reportErrorIfAny(cuCtxCreate(&cu_ctx, 0, device), "cuCtxCreate");

  gpu::BaseCudaContextOption opt;
  cuStreamCreate(&opt.stream, CU_STREAM_DEFAULT);
  gpu::BaseCudaContext context(opt);

  // bind inputs
  std::vector<void*> h_data(num_inputs);
  for (int idx = 0; idx < num_inputs; ++idx) {
    const buffer_shape_t& shape = input_shapes[idx];
    const ElemType& dtype = input_elem_types[idx];
    int64_t nelem = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      nelem *= shape[i];
    }
    if (dtype == ElemType::kF32) {
      int64_t bytes = nelem * sizeof(float);
      h_data[idx] = new float[nelem];
      for (size_t i = 0; i < nelem; ++i) {
        reinterpret_cast<float*>(h_data[idx])[i] = 1.0 + i;
      }
      if (input_placement[idx]) {
        // input on host memory
        context.bindInput(idx, h_data[idx], shape);
      } else {
        // input on device memory
        float* d_addr = nullptr;
        reportErrorIfAny(cuMemAlloc((CUdeviceptr*)&d_addr, bytes),
                         "cuMemAlloc");
        reportErrorIfAny(cuMemcpyHtoD((CUdeviceptr)d_addr, h_data[idx], bytes),
                         "cuMemcpyHtoD");
        context.bindInput(idx, d_addr, shape);
      }
    } else if (dtype == ElemType::kI32) {
      int64_t bytes = nelem * sizeof(int32_t);
      h_data[idx] = new int32_t[nelem];
      for (size_t i = 0; i < nelem; ++i) {
        reinterpret_cast<int32_t*>(h_data[idx])[i] = 1 + i;
      }
      if (input_placement[idx]) {
        // input on host memory
        std::cout << "host data ptr to be binded: " << h_data[idx] << std::endl;
        context.bindInput(idx, h_data[idx], shape);
      } else {
        // input on device memory
        int32_t* d_addr = nullptr;
        reportErrorIfAny(cuMemAlloc((CUdeviceptr*)&d_addr, bytes),
                         "cuMemAlloc");
        reportErrorIfAny(cuMemcpyHtoD((CUdeviceptr)d_addr, h_data[idx], bytes),
                         "cuMemcpyHtoD");
        std::cout << "device data ptr to be binded: " << d_addr << std::endl;
        context.bindInput(idx, d_addr, shape);
      }
    } else if (dtype == ElemType::kI1) {
      int64_t bytes = nelem * sizeof(bool);
      h_data[idx] = new bool[nelem];
      for (size_t i = 0; i < nelem; ++i) {
        reinterpret_cast<bool*>(h_data[idx])[i] = (1 + i) % 2;
      }
      if (input_placement[idx]) {
        // input on host memory
        std::cout << "host data ptr to be binded: " << h_data[idx] << std::endl;
        context.bindInput(idx, h_data[idx], shape);
      } else {
        // input on device memory
        int32_t* d_addr = nullptr;
        reportErrorIfAny(cuMemAlloc((CUdeviceptr*)&d_addr, bytes),
                         "cuMemAlloc");
        reportErrorIfAny(cuMemcpyHtoD((CUdeviceptr)d_addr, h_data[idx], bytes),
                         "cuMemcpyHtoD");
        std::cout << "device data ptr to be binded: " << d_addr << std::endl;
        context.bindInput(idx, d_addr, shape);
      }
    } else {
      std::cout << "unexpected input dtype\n";
      return 1;
    }
  }

  context.onExecutionStart();

  void* ctx_struct[] = {&context, tao_ral_func_ptr};
  void* ral_ctx_ptr = (void*)(&ctx_struct);
  std::cout << "######### tao_ctx: " << ral_ctx_ptr << std::endl;
  void* args[1] = {(void*)&ral_ctx_ptr};
  entry_func(args);

  // bind outputs
  std::vector<buffer_shape_t> output_shapes;
  std::vector<void*> d_results;
  for (int idx = 0; idx < num_outputs; ++idx) {
    void* d_result = nullptr;
    buffer_shape_t out_shape;
    context.bindOutput(idx, (void**)&d_result, &out_shape);
    output_shapes.emplace_back(out_shape);
    d_results.emplace_back(d_result);
  }

  context.onExecutionFinish();

  for (int idx = 0; idx < num_outputs; ++idx) {
    const buffer_shape_t shape = output_shapes[idx];
    int64_t nelem = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      nelem *= shape[i];
    }
    void* d_result = d_results[idx];
    print_output_shape(d_result, shape);
    if (out_elem_types[idx] == ElemType::kF32) {
      if (output_placement[idx]) {
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": "
                    << reinterpret_cast<float*>(d_result)[i] << std::endl;
        }
      } else {
        int64_t bytes = nelem * sizeof(float);
        float h_result[nelem];
        reportErrorIfAny(
            cuMemcpyDtoH((void*)h_result,
                         reinterpret_cast<CUdeviceptr>(d_result), bytes),
            "cuMemcpyDtoH");
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": " << h_result[i] << std::endl;
        }
      }
    } else if (out_elem_types[idx] == ElemType::kI32) {
      if (output_placement[idx]) {
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": "
                    << reinterpret_cast<int32_t*>(d_result)[i] << std::endl;
        }
      } else {
        int64_t bytes = nelem * sizeof(int32_t);
        int32_t h_result[nelem];
        reportErrorIfAny(
            cuMemcpyDtoH((void*)h_result,
                         reinterpret_cast<CUdeviceptr>(d_result), bytes),
            "cuMemcpyDtoH");
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": " << h_result[i] << std::endl;
        }
      }
    } else if (out_elem_types[idx] == ElemType::kI1) {
      if (output_placement[idx]) {
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": "
                    << reinterpret_cast<bool*>(d_result)[i] << std::endl;
        }
      } else {
        int64_t bytes = nelem * sizeof(bool);
        bool h_result[nelem];
        reportErrorIfAny(
            cuMemcpyDtoH((void*)h_result,
                         reinterpret_cast<CUdeviceptr>(d_result), bytes),
            "cuMemcpyDtoH");
        for (int i = 0; i < nelem; ++i) {
          std::cout << "\tresult #" << i << ": " << h_result[i] << std::endl;
        }
      }
    } else {
      std::cout << "unsupported output element type" << std::endl;
      return 1;
    }
  }

  for (void* p : h_data) {
    delete[] p;
  }

  return 0;
}
