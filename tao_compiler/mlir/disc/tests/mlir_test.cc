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

#include "mlir/disc/tests/mlir_test.h"

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

#if GOOGLE_CUDA
#include <cuda_profiler_api.h>
#endif

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "mlir/xla/ral/context/base/cuda/cuda_context_impl.h"
#else
// TODO(disc): figure out why the bazel does not trigger re-compile this file
// after we update ral.
//
#include "mlir/xla/ral/context/base/cpu/cpu_context_impl.h"
#endif

#include "mlir/xla/ral/ral_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir_test {

using ::Eigen::half;
using tensorflow::DataType;
using tensorflow::ReadStringFromEnvVar;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::Internal;
constexpr int c_ProfilingWarmUpSteps = 50;

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuStatus;
#endif

#if TENSORFLOW_USE_ROCM
#define GPU_SUCCESS hipSuccess
#define GPU_MEMCPYDTOH_API tensorflow::wrap::hipMemcpyDtoH
#define GPU_MEMCPYHTOD_API tensorflow::wrap::hipMemcpyHtoD
#define GPU_MALLOC_API tensorflow::wrap::hipMalloc
#define GPU_FREE_API tensorflow::wrap::hipFree
#else
#define GPU_SUCCESS CUDA_SUCCESS
#define GPU_MEMCPYDTOH_API cuMemcpyDtoH
#define GPU_MEMCPYHTOD_API cuMemcpyHtoD
#define GPU_MALLOC_API cuMemAlloc
#define GPU_FREE_API cuMemFree
#endif

DataType ParseDataType(const std::string& s) {
  if (s == "f32") {
    return tensorflow::DT_FLOAT;
  } else if (s == "f64") {
    return tensorflow::DT_DOUBLE;
  } else if (s == "f16") {
    return tensorflow::DT_HALF;
  } else if (s == "i32") {
    return tensorflow::DT_INT32;
  } else if (s == "i64") {
    return tensorflow::DT_INT64;
  } else if (s == "i1") {
    return tensorflow::DT_BOOL;
  } else if (s == "ui8") {
    return tensorflow::DT_UINT8;
  } else if (s == "i8") {
    return tensorflow::DT_INT8;
  } else if (s == "qi8") {
    return tensorflow::DT_QINT8;
  } else if (s == "qui8") {
    return tensorflow::DT_QUINT8;
  } else if (s == "qi32") {
    return tensorflow::DT_QINT32;
  } else {
    LOG(ERROR) << "Error: unsupported input/output element type " << s;
    return tensorflow::DT_INVALID;
  }
}

DeviceType getDefaultPlacementForBackend(const BackendType& backend) {
  if (backend == BackendType::kCuda) {
    return DeviceType::kGPU;
  } else if (backend == BackendType::kX86) {
    return DeviceType::kCPU;
  } else if (backend == BackendType::kAArch64) {
    return DeviceType::kCPU;
  } else {
    LOG(FATAL) << "Unrecognized backend type";
    return DeviceType::kCPU;
  }
}

buffer_shape_t ParseInputDescriptor(const std::string& s,
                                    const BackendType& backend, DataType* dtype,
                                    DeviceType* placement) {
  buffer_shape_t shape;
  std::vector<std::string> splitted = absl::StrSplit(s, 'x');
  for (int i = 0; i < splitted.size() - 1; ++i) {
    shape.emplace_back(std::stoi(splitted[i]));
  }
  std::vector<std::string> dtype_device = absl::StrSplit(splitted.back(), '_');
  *dtype = ParseDataType(dtype_device.front());
  if (dtype_device.back() == "h") {
    *placement = DeviceType::kCPU;
  } else if (dtype_device.back() == "d") {
    *placement = DeviceType::kGPU;
  } else if (dtype_device.back() == "X") {
    // 'X' means using default placement according to the backend type
    *placement = getDefaultPlacementForBackend(backend);
  } else {
    LOG(ERROR) << "Error: placement of input tensor not properly assigned:"
               << " format is like: 2x3xf32_{h|d|X}";
  }
  return shape;
}

void ParseOutputDescriptor(const std::string& s, const BackendType& backend,
                           DataType* dtype, DeviceType* placement) {
  std::vector<std::string> splitted = absl::StrSplit(s, '_');
  *dtype = ParseDataType(splitted.front());
  if (splitted.back() == "h") {
    *placement = DeviceType::kCPU;
  } else if (splitted.back() == "d") {
    *placement = DeviceType::kGPU;
  } else if (splitted.back() == "X") {
    // 'X' means using default placement according to the backend type
    *placement = getDefaultPlacementForBackend(backend);
  } else {
    LOG(ERROR) << "Error: placement of output tensor not properly assigned:"
               << " format is like: f32_{h|d|X}";
  }
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
static void printErrorIfAny(GpuStatus result, const char* where) {
  if (result != GPU_SUCCESS) {
    std::ostringstream out;
    LOG(ERROR) << "CUDA failed with " << result << " in " << where;
  }
}
static int32_t reportErrorIfAny(GpuStatus result, const char* where) {
  printErrorIfAny(result, where);
  return result;
}
static void gpu_dealloc(void* buffer) {
#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(GPU_FREE_API(absl::bit_cast<hipDeviceptr_t>(buffer)),
                   "hipFree");
#else
  reportErrorIfAny(GPU_FREE_API(CUdeviceptr(buffer)), "cuMemFree");
#endif
}
#endif

void print_output_shape(void* d_result, const buffer_shape_t& shape) {
  VLOG(0) << "out buffer = " << d_result;
  VLOG(0) << "out shape:";
  for (size_t i = 0; i < shape.size(); ++i) {
    VLOG(0) << "  dim #" << i << ": " << shape[i];
  }
}

MlirTest::MlirTest(const std::string& mlir_file_path,
                   const std::string& tmp_dir, const std::string& test_name,
                   int num_inputs, int num_outputs,
                   const std::vector<buffer_shape_t>& input_shapes,
                   const std::vector<DataType>& input_elem_types,
                   const std::vector<DeviceType>& input_placement,
                   const std::vector<std::vector<float>>& input_vals,
                   const std::vector<DataType>& out_elem_types,
                   const std::vector<DeviceType>& output_placement,
                   const std::vector<tensorflow::Tensor>& expected_output_vals,
                   bool profiling, bool multi_cc_mode,
                   bool multi_cc_mode_dbg_ptx_only)
    : mlir_file_path_(mlir_file_path),
      tmp_dir_(tmp_dir),
      test_name_(test_name),
      num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      input_shapes_(input_shapes),
      input_elem_types_(input_elem_types),
      input_placement_(input_placement),
      input_vals_(input_vals),
      out_elem_types_(out_elem_types),
      output_placement_(output_placement),
      expected_output_vals_(expected_output_vals),
      h_data_(num_inputs),
      actual_results_(num_outputs),
      profiling_(profiling),
      multi_cc_mode_(multi_cc_mode),
      multi_cc_mode_dbg_ptx_only_(multi_cc_mode_dbg_ptx_only) {
  ReadStringFromEnvVar(
      "TF_OPT_PATH", "external/org_tensorflow/tensorflow/compiler/mlir/tf-opt",
      &tf_opt_path_);

  ReadStringFromEnvVar("DHLO_COMPILER_MAIN_PATH",
                       "mlir/disc/disc_compiler_main",
                       &dhlo_compiler_main_path_);

  ReadStringFromEnvVar(
      "TF_MLIR_TRANSLATE_PATH",
      "external/org_tensorflow/tensorflow/compiler/mlir/tf-mlir-translate",
      &tf_mlir_translate_path_);

  compiled_so_file_ = tmp_dir_ + test_name_ + ".so";
  VLOG(0) << "tf_opt_pat: " << tf_opt_path_;
  VLOG(0) << "mlir_file_path: " << mlir_file_path_;
  VLOG(0) << "tmp_dir: " << tmp_dir_;
  VLOG(0) << "test_name: " << test_name_;
}

Status MlirTest::Run() {
  TF_RETURN_IF_ERROR(CompileMlirToBinary());
  VLOG(0) << "run compiled program\n";
  TF_RETURN_IF_ERROR(GenerateInputAndRun());
  if (expected_output_vals_.empty()) {
    VLOG(0) << "run golden tf\n";
    TF_RETURN_IF_ERROR(RunGoldenTF());
  }
  TF_RETURN_IF_ERROR(CompareResults());
  return tsl::OkStatus();
}

int MlirTest::CallBinary(std::string program_path,
                         std::vector<std::string> args) {
  tensorflow::SubProcess process;
  process.SetProgram(program_path, args);
  process.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  process.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);
  VLOG(0) << "program_path: " << program_path << "\n";
  if (!process.Start()) {
    return 1;
  }
  std::string stdout_output;
  std::string stderr_output;
  int s = process.Communicate(
      /*stdin_input=*/nullptr, &stdout_output, &stderr_output);
  std::string msg = "Executed: ";
  for (auto arg : args) {
    absl::StrAppend(&msg, arg, " ");
  }
  VLOG(0) << msg;
  VLOG(0) << program_path << ": " << s;
  VLOG(0) << "-- stdout:\n"
          << stdout_output << "\n============ END ============\n";
  VLOG(0) << "-- stderr:\n"
          << stderr_output << "\n============ END ============\n";
  VLOG(0) << "ret: " << s << "\n";
  return s;
}

Status MlirTest::CompileMlirToBinary() {
  // tf executor dialect -> tf dialect
  std::string tf_dialect_file = tmp_dir_ + test_name_ + "_tf_dialect.mlir";
  std::vector<std::string> args = {tf_opt_path_, "--tf-standard-pipeline",
                                   mlir_file_path_, "-o", tf_dialect_file};
  VLOG(0) << "tf_opt_path: " << tf_opt_path_ << "\n";
  if (CallBinary(tf_opt_path_, args)) {
    return Internal("tf_executor dialect -> tf dialect failed");
  }

  // tf dialect -> out.so
  setenv("TAO_MLIR_DUMP", "false", 1);
  auto allow_hex = getenv("TAO_MLIR_ALLOW_HEX");
  std::string allow_hex_str = allow_hex ? allow_hex : "-1";
  args = {dhlo_compiler_main_path_,
          "--mlir-print-elementsattrs-with-hex-if-larger",
          allow_hex_str,
          "--mlir-elide-elementsattrs-if-larger",
          "8",
          tf_dialect_file,
          compiled_so_file_};
  if (multi_cc_mode_) {
    args.emplace_back("--multi-cc-support");
  }
  if (multi_cc_mode_dbg_ptx_only_) {
    args.emplace_back("--multi-cc-support-dbg-ptx-only");
  }
  if (CallBinary(dhlo_compiler_main_path_, args)) {
    return Internal("tf dialect -> compilation result failed");
  }

  return tsl::OkStatus();
}

Status MlirTest::LoadGraph(const std::string& graph_file_name) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadTextProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return Internal("Error: read pb file failed");
  }

  for (auto& node_def : graph_def.node()) {
    if (node_def.op() == "_Retval") {
      output_tensor_name_map_[node_def.name()] = node_def.input(0);
    }
  }

  tensorflow::SessionOptions options;
  options.config.mutable_gpu_options()->set_allow_growth(true);
  sess_.reset(tensorflow::NewSession(options));
  Status session_create_status = sess_->Create(graph_def);
  if (!session_create_status.ok()) {
    return Internal("Error: create session failed" +
                    session_create_status.error_message());
  }
  return tsl::OkStatus();
}

// TODO: configurable relative/absolute error tolerance according to
// different pattern under test.
template <class T>
bool MlirTest::IsAcceptableNear(T a, T b, double rel_err_limit,
                                double abs_err_limit) {
  if (a == b) return true;
  auto a_cast = static_cast<double>(a);
  auto b_cast = static_cast<double>(b);
  if (std::isnan(a_cast) && std::isnan(b_cast)) return true;
  double abs_err = std::abs(a_cast - b_cast);
  double rel_err =
      std::abs(a_cast - b_cast) / std::max(abs(a_cast), abs(b_cast));
  return abs_err < abs_err_limit || rel_err < rel_err_limit;
}

template <class T>
static void InitializeTensor(const std::vector<T>& initialization_values,
                             Tensor* input_tensor) {
  auto type_tensor =
      input_tensor->bit_casted_shaped<T, 1>({input_tensor->NumElements()});
  type_tensor = type_tensor.constant(static_cast<T>(0));
  if (!initialization_values.empty()) {
    for (int i = 0; i < initialization_values.size(); ++i) {
      type_tensor(i) = static_cast<T>(initialization_values[i]);
    }
  }
}

// tf_executor dialect -> graphdef
Status MlirTest::RunGoldenTF() {
  // // TODO: This as a workaround required by the current version of
  // // tf-mlir-translator, which is to insert a Placeholder.input for each
  // input
  // // argument. To be removed after later porting. tf-opt
  // // --tf-executor-insert-placeholder xx.mlir -o ph_xx.mlir
  // std::string mlir_with_ph_path = tmp_dir_ + test_name_ + "_ph.mlir";
  // std::vector<std::string> args = {tf_opt_path_,
  //                                  "--tf-executor-insert-placeholder",
  //                                  mlir_file_path_, "-o", mlir_with_ph_path};
  // if (CallBinary(tf_opt_path_, args)) {
  //   return Internal("tf-executor-insert-placeholder failed.");
  // }
  std::vector<std::string> args;
  std::string mlir_with_ph_path = mlir_file_path_;

  // tf-mlir-translate -mlir-to-graphdef xx.mlir -o xx.mlir.pbtxt
  std::string graphdef_path = tmp_dir_ + test_name_ + ".pbtxt";
  args = {tf_mlir_translate_path_, "-mlir-to-graphdef", mlir_with_ph_path, "-o",
          graphdef_path};
  if (CallBinary(tf_mlir_translate_path_, args)) {
    return Internal("mlir-to-graphdef failed.");
  }

  VLOG(0) << "graphdef_path: " << graphdef_path;
  TF_RETURN_IF_ERROR(LoadGraph(graphdef_path));

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  for (int64_t i = 0; i < num_inputs_; ++i) {
    TensorShape input_shape;
    auto input_shape_vec = input_shapes_[i];
    int64_t num_elements = 1;
    for (int64_t dim = 0; dim < input_shape_vec.size(); ++dim) {
      auto dim_size = input_shape_vec[dim];
      input_shape.AddDim(dim_size);
      num_elements *= dim_size;
    }
    DataType dtype = input_elem_types_[i];
    Tensor input_tensor(dtype, input_shape);
    if (dtype == tensorflow::DT_FLOAT) {
      float* ptr = reinterpret_cast<float*>(h_data_[i].get());
      std::vector<float> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<float>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_DOUBLE) {
      double* ptr = reinterpret_cast<double*>(h_data_[i].get());
      std::vector<double> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<double>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_HALF) {
      half* ptr = reinterpret_cast<half*>(h_data_[i].get());
      std::vector<half> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<half>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_INT32 ||
               dtype == tensorflow::DT_QINT32) {
      int32_t* ptr = reinterpret_cast<int32_t*>(h_data_[i].get());
      std::vector<int32_t> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<int32_t>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_INT64) {
      tensorflow::int64* ptr =
          reinterpret_cast<tensorflow::int64*>(h_data_[i].get());
      std::vector<tensorflow::int64> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<tensorflow::int64>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_BOOL) {
      bool* ptr = reinterpret_cast<bool*>(h_data_[i].get());
      std::vector<bool> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<bool>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_UINT8 ||
               dtype == tensorflow::DT_QUINT8) {
      uint8_t* ptr = reinterpret_cast<uint8_t*>(h_data_[i].get());
      std::vector<uint8_t> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<uint8_t>(h_data_vec, &input_tensor);
    } else if (dtype == tensorflow::DT_INT8 || dtype == tensorflow::DT_QINT8) {
      int8_t* ptr = reinterpret_cast<int8_t*>(h_data_[i].get());
      std::vector<int8_t> h_data_vec(ptr, ptr + num_elements);
      InitializeTensor<int8_t>(h_data_vec, &input_tensor);
    } else {
      return Internal("unexpected datatype");
    }
    std::string name = "input";
    absl::StrAppend(&name, i);
    input_tensors.emplace_back(std::make_pair(name, input_tensor));
  }
  std::vector<std::string> output_tensor_names;
  std::vector<Tensor> output_tensors;
  for (int64_t i = 0; i < num_outputs_; ++i) {
    std::string name = "output";
    absl::StrAppend(&name, i);
    output_tensor_names.emplace_back(output_tensor_name_map_[name]);
  }

  for (int iter = 0; iter < 5; ++iter) {
    output_tensors.clear();
    std::chrono::steady_clock::time_point tf_begin =
        std::chrono::steady_clock::now();

    TF_RETURN_IF_ERROR(
        sess_->Run(input_tensors, output_tensor_names, {}, &output_tensors));

    std::chrono::steady_clock::time_point tf_end =
        std::chrono::steady_clock::now();

    VLOG(0) << "--- TF Execution uses: "
            << (std::chrono::duration_cast<std::chrono::microseconds>(tf_end -
                                                                      tf_begin)
                    .count()) /
                   1000.0
            << " ms";
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      VLOG(0) << "\toutput shape #" << i << ": " << output_tensors[i].shape();
    }
  }

  expected_output_vals_ = std::move(output_tensors);
  return tsl::OkStatus();
}

Status MlirTest::CompareResults() {
  for (int64_t i = 0; i < num_outputs_; ++i) {
    VLOG(0) << "processing output " << i;
    DataType dtype = out_elem_types_[i];
    std::string msg = "Error in output ";
    absl::StrAppend(&msg, i, ":\n");
    if (dtype == tensorflow::DT_FLOAT) {
      auto datas = expected_output_vals_[i].flat<float>();
      for (int64_t n = 0; n < datas.size(); ++n) {
        float actual = reinterpret_cast<float*>(actual_results_[i].get())[n];
        VLOG(2) << "  expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual)) {
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_DOUBLE) {
      auto datas = expected_output_vals_[i].flat<double>();
      for (int64_t n = 0; n < datas.size(); ++n) {
        double actual = reinterpret_cast<double*>(actual_results_[i].get())[n];
        VLOG(2) << "  expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual, 5e-2, 1e-3)) {
          absl::StrAppend(&msg, i, " index: ", n,
                          " expected: ", static_cast<double>(datas(n)),
                          ", actual: ", static_cast<double>(actual), "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_HALF) {
      auto datas = expected_output_vals_[i].flat<half>();
      for (int64_t n = 0; n < datas.size(); ++n) {
        half actual = reinterpret_cast<half*>(actual_results_[i].get())[n];
        VLOG(2) << "  expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual, 5e-2, 1e-3)) {
          absl::StrAppend(&msg, i, " index: ", n,
                          " expected: ", static_cast<float>(datas(n)),
                          ", actual: ", static_cast<float>(actual), "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_INT32 ||
               dtype == tensorflow::DT_QINT32) {
      // Using bitcast instead of `flat` in case the output has quantized
      // integer type.
      auto datas = expected_output_vals_[i].bit_casted_shaped<int32_t, 1>(
          {expected_output_vals_[i].NumElements()});
      for (int64_t n = 0; n < datas.size(); ++n) {
        int32_t actual =
            reinterpret_cast<int32_t*>(actual_results_[i].get())[n];
        VLOG(2) << "expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual)) {
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_INT64) {
      auto datas = expected_output_vals_[i].flat<tensorflow::int64>();
      for (int64_t n = 0; n < datas.size(); ++n) {
        tensorflow::int64 actual =
            reinterpret_cast<int64_t*>(actual_results_[i].get())[n];
        VLOG(2) << "expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual)) {
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_BOOL) {
      auto datas = expected_output_vals_[i].flat<bool>();
      for (int64_t n = 0; n < datas.size(); ++n) {
        bool actual = reinterpret_cast<bool*>(actual_results_[i].get())[n];
        VLOG(2) << "expected: " << datas(n) << ", actual: " << actual;
        if (datas(n) != actual) {
          std::string msg = "Error in output ";
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_UINT8 ||
               dtype == tensorflow::DT_QUINT8) {
      // Using bitcast instead of `flat` in case the output has quantized
      // integer type.
      auto datas = expected_output_vals_[i].bit_casted_shaped<uint8_t, 1>(
          {expected_output_vals_[i].NumElements()});
      for (int64_t n = 0; n < datas.size(); ++n) {
        uint8_t actual =
            reinterpret_cast<uint8_t*>(actual_results_[i].get())[n];
        VLOG(2) << "expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual)) {
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else if (dtype == tensorflow::DT_INT8 || dtype == tensorflow::DT_QINT8) {
      // Using bitcast instead of `flat` in case the output has quantized
      // integer type.
      auto datas = expected_output_vals_[i].bit_casted_shaped<int8_t, 1>(
          {expected_output_vals_[i].NumElements()});
      for (int64_t n = 0; n < datas.size(); ++n) {
        int8_t actual = reinterpret_cast<int8_t*>(actual_results_[i].get())[n];
        VLOG(2) << "expected: " << datas(n) << ", actual: " << actual;
        if (!MlirTest::IsAcceptableNear(datas(n), actual)) {
          absl::StrAppend(&msg, i, " index: ", n, " expected: ", datas(n),
                          ", actual: ", actual, "\n");
          return Internal(msg);
        }
      }
    } else {
      return Internal("Error: unexpected output tensor dtype");
    }
  }
  return tsl::OkStatus();
}

MlirTestImpl::MlirTestImpl(
    const std::string& mlir_file_path, const std::string& tmp_dir,
    const std::string& test_name, int num_inputs, int num_outputs,
    const std::vector<buffer_shape_t>& input_shapes,
    const std::vector<DataType>& input_elem_types,
    const std::vector<DeviceType>& input_placement,
    const std::vector<std::vector<float>>& input_vals,
    const std::vector<DataType>& out_elem_types,
    const std::vector<DeviceType>& output_placement,
    const std::vector<tensorflow::Tensor>& expected_output_vals, bool profiling,
    bool multi_cc_mode, bool multi_cc_mode_dbg_ptx_only)
    : MlirTest(mlir_file_path, tmp_dir, test_name, num_inputs, num_outputs,
               input_shapes, input_elem_types, input_placement, input_vals,
               out_elem_types, output_placement, expected_output_vals,
               profiling, multi_cc_mode, multi_cc_mode_dbg_ptx_only) {
  tao_ral_func_ptr_ = reinterpret_cast<void*>(&tao_ral_call_impl);
  if (!tao_ral_func_ptr_) {
    LOG(ERROR) << "Error: fail to find tao_ral_call_impl";
  }
  std::srand(43);
}

MlirTestImpl::~MlirTestImpl() { output_buffers_.clear(); }

Status MlirTestImpl::GenerateInputAndRun() {
  void* func_handle = dlopen(compiled_so_file_.c_str(), RTLD_NOW);
  if (!func_handle) {
    std::string msg = "fail to open compiled .so file with error: ";
    absl::StrAppend(&msg, dlerror());
    return Internal(msg);
  }

  void* entry_func_ptr = dlsym(func_handle, "main");
  if (!entry_func_ptr) {
    return Internal("fail to find main");
  }
  using func_t = void (*)(void**);
  func_t entry_func = (func_t)entry_func_ptr;

  tao::ral::BaseContextOption opt;
  opt.metadata_file_path = compiled_so_file_ + ".pbtxt";
  opt.cache_workspace_mem_across_execution = true;
  tao::ral::cpu::BaseCpuContextOption cpu_opt;
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.use_stream_executor = true;
  context_ = tao::ral::gpu::MakeBaseCudaContext(opt, cpu_opt, gpu_opt);
#else
  context_ = tao::ral::cpu::MakeBaseCpuContext(opt, cpu_opt);
#endif

  // bind inputs
  for (int idx = 0; idx < num_inputs_; ++idx) {
    const buffer_shape_t& shape = input_shapes_[idx];
    const DataType& dtype = input_elem_types_[idx];
    int64_t nelem = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      nelem *= shape[i];
    }
    nelem = (nelem ? nelem : 1);
    int64_t bytes = -1;
    void* d_addr = nullptr;
    if (dtype == tensorflow::DT_FLOAT) {
      bytes = nelem * sizeof(float);
      h_data_[idx] = std::shared_ptr<void>(new float[nelem], [](void* p) {
        delete[] reinterpret_cast<float*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<float*>(h_data_[idx].get())[i] = 1.0 + i;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<float*>(h_data_[idx].get())[i] = input_vals_[idx][m];
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_DOUBLE) {
      bytes = nelem * sizeof(double);
      h_data_[idx] = std::shared_ptr<void>(new double[nelem], [](void* p) {
        delete[] reinterpret_cast<double*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          float v = std::rand() % 1000 / 1000.0;
          reinterpret_cast<double*>(h_data_[idx].get())[i] =
              static_cast<double>(v);
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<double*>(h_data_[idx].get())[i] =
              static_cast<double>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_HALF) {
      bytes = nelem * sizeof(half);
      h_data_[idx] = std::shared_ptr<void>(new half[nelem], [](void* p) {
        delete[] reinterpret_cast<half*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          float v = std::rand() % 1000 / 1000.0;
          reinterpret_cast<half*>(h_data_[idx].get())[i] = static_cast<half>(v);
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<half*>(h_data_[idx].get())[i] =
              static_cast<half>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_INT32 ||
               dtype == tensorflow::DT_QINT32) {
      bytes = nelem * sizeof(int32_t);
      h_data_[idx] = std::shared_ptr<void>(new int32_t[nelem], [](void* p) {
        delete[] reinterpret_cast<int32_t*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int32_t*>(h_data_[idx].get())[i] = 1 + i;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int32_t*>(h_data_[idx].get())[i] =
              static_cast<int32_t>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_INT64) {
      bytes = nelem * sizeof(int64_t);
      h_data_[idx] = std::shared_ptr<void>(new int64_t[nelem], [](void* p) {
        delete[] reinterpret_cast<int64_t*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int64_t*>(h_data_[idx].get())[i] = 1 + i;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int64_t*>(h_data_[idx].get())[i] =
              static_cast<int64_t>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_BOOL) {
      bytes = nelem * sizeof(bool);
      h_data_[idx] = std::shared_ptr<void>(new bool[nelem], [](void* p) {
        delete[] reinterpret_cast<bool*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<bool*>(h_data_[idx].get())[i] = true;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<bool*>(h_data_[idx].get())[i] =
              static_cast<bool>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }

    } else if (dtype == tensorflow::DT_UINT8 ||
               dtype == tensorflow::DT_QUINT8) {
      bytes = nelem * sizeof(uint8_t);
      h_data_[idx] = std::shared_ptr<void>(new uint8_t[nelem], [](void* p) {
        delete[] reinterpret_cast<uint8_t*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<uint8_t*>(h_data_[idx].get())[i] = 1 + i;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<uint8_t*>(h_data_[idx].get())[i] =
              static_cast<uint8_t>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }
    } else if (dtype == tensorflow::DT_INT8 || dtype == tensorflow::DT_QINT8) {
      bytes = nelem * sizeof(int8_t);
      h_data_[idx] = std::shared_ptr<void>(new int8_t[nelem], [](void* p) {
        delete[] reinterpret_cast<int8_t*>(p);
      });
      if (input_vals_.empty() || input_vals_[idx].empty()) {
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int8_t*>(h_data_[idx].get())[i] = 1 + i;
        }
      } else {
        size_t m = 0;
        for (size_t i = 0; i < nelem; ++i) {
          reinterpret_cast<int8_t*>(h_data_[idx].get())[i] =
              static_cast<int8_t>(input_vals_[idx][m]);
          m = (m + 1) % nelem;
        }
      }
    } else {
      return Internal("unexpected input dtype");
    }
  }

  std::vector<buffer_shape_t> output_shapes;
  output_buffers_.resize(num_outputs_);

  std::vector<void*> d_addr_vec(num_inputs_);
  std::vector<int64_t> bytes_vec(num_inputs_);
  for (int idx = 0; idx < num_inputs_; ++idx) {
    const buffer_shape_t& shape = input_shapes_[idx];
    const DataType& dtype = input_elem_types_[idx];
    int64_t nelem = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      nelem *= shape[i];
    }
    nelem = (nelem ? nelem : 1);
    int64_t bytes = -1;
    if (dtype == tensorflow::DT_FLOAT) {
      bytes = nelem * sizeof(float);
    } else if (dtype == tensorflow::DT_DOUBLE) {
      bytes = nelem * sizeof(double);
    } else if (dtype == tensorflow::DT_HALF) {
      bytes = nelem * sizeof(half);
    } else if (dtype == tensorflow::DT_INT32 ||
               dtype == tensorflow::DT_QINT32) {
      bytes = nelem * sizeof(int32_t);
    } else if (dtype == tensorflow::DT_INT64) {
      bytes = nelem * sizeof(int64_t);
    } else if (dtype == tensorflow::DT_BOOL) {
      bytes = nelem * sizeof(bool);
    } else if (dtype == tensorflow::DT_UINT8 ||
               dtype == tensorflow::DT_QUINT8) {
      bytes = nelem * sizeof(uint8_t);
    } else if (dtype == tensorflow::DT_INT8 || dtype == tensorflow::DT_QINT8) {
      bytes = nelem * sizeof(int8_t);
    }
    void* d_addr = nullptr;
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    if (input_placement_[idx] != DeviceType::kCPU) {
      // input on device memory
      reportErrorIfAny(GPU_MALLOC_API((GpuDevicePtr*)&d_addr, bytes),
                       "GPU Malloc");
      d_addr_vec[idx] = d_addr;
      bytes_vec[idx] = bytes;
    }
#elif TAO_CPU_ONLY
    if (input_placement_[idx] != DeviceType::kCPU) {
      return Internal(
          "unexpected input placement, expect host tag for cpu only build");
    }
#endif
  }

  int num_iters = profiling_ ? c_ProfilingWarmUpSteps : 1;
  for (int iter = 0; iter < num_iters; iter++) {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    {
#if GOOGLE_CUDA
      if (profiling_ && (iter == num_iters - 1)) {
        cudaProfilerStart();
      }
#endif
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
      auto exec_ctx = tao::ral::MakeExecutionContext<
          tao::ral::gpu::BaseCudaExecutionContext>(context_.get());
#else
      auto exec_ctx = tao::ral::MakeExecutionContext<
          tao::ral::cpu::BaseCpuExecutionContext>(context_.get());
#endif

      for (int idx = 0; idx < num_inputs_; ++idx) {
        const buffer_shape_t& shape = input_shapes_[idx];
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
        if (input_placement_[idx] == DeviceType::kCPU) {
          // input on host memory
          exec_ctx->bindInput(idx, h_data_[idx].get(), shape);
        } else {
          // input on device memory
          void* d_addr = d_addr_vec[idx];
          reportErrorIfAny(
              GPU_MEMCPYHTOD_API((GpuDevicePtr)d_addr, h_data_[idx].get(),
                                 bytes_vec[idx]),
              "GPU MemcpyHtoD");
          exec_ctx->bindInput(idx, d_addr, shape);
        }
#else
        exec_ctx->bindInput(idx, h_data_[idx].get(), shape);
#endif
      }

      void* ctx_struct[] = {exec_ctx.get(), tao_ral_func_ptr_};
      void** args = (void**)(&ctx_struct);
      // VLOG(2) << "######### tao_ctx: " << ral_ctx_ptr;
      // void* args[1] = {(void*)&ral_ctx_ptr};
      entry_func(args);

      // bind outputs
      output_shapes.clear();
      for (int idx = 0; idx < num_outputs_; ++idx) {
        output_buffers_[idx].reset();
        exec_ctx->bindOutput(idx, &output_buffers_[idx]);
        output_shapes.emplace_back(output_buffers_[idx]->shape());
        void* result = (void*)output_buffers_[idx]->data();
        // if the output tensor is on device, temporarily store device address,
        // and will be replaced by host address later after potential memcpy.
        actual_results_[idx] = std::shared_ptr<void>(result, [](void* p) {
          // do nothing
        });
      }
    }

#if GOOGLE_CUDA
    if (profiling_ && (iter == num_iters - 1)) {
      cudaProfilerStop();
    }
#endif

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    VLOG(0) << "--- MLIR Execution uses: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                       .count() /
                   1000.0
            << " ms";
  }

  for (int idx = 0; idx < num_outputs_; ++idx) {
    const buffer_shape_t shape = output_shapes[idx];
    int64_t nelem = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      nelem *= shape[i];
    }
    void* result = actual_results_[idx].get();
    print_output_shape(result, shape);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    if (out_elem_types_[idx] == tensorflow::DT_FLOAT) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<float*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(float);
        float* h_result = nullptr;
        if (nelem) {
          h_result = new float[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<float*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_DOUBLE) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<double*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(double);
        double* h_result = nullptr;
        if (nelem) {
          h_result = new double[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<double*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_HALF) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<half*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(half);
        half* h_result = nullptr;
        if (nelem) {
          h_result = new half[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<half*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT32 ||
               out_elem_types_[idx] == tensorflow::DT_QINT32) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<int32_t*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(int32_t);
        int* h_result = nullptr;
        if (nelem) {
          h_result = new int[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<int32_t*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT64) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<int64_t*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(int64_t);
        int64_t* h_result = nullptr;
        if (nelem) {
          h_result = new int64_t[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<int64_t*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_BOOL) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<bool*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(bool);
        bool* h_result = nullptr;
        if (nelem) {
          h_result = new bool[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<bool*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_UINT8 ||
               out_elem_types_[idx] == tensorflow::DT_QUINT8) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<uint8_t*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(uint8_t);
        int* h_result = nullptr;
        if (nelem) {
          h_result = new int[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<uint8_t*>(p); });
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT8 ||
               out_elem_types_[idx] == tensorflow::DT_QINT8) {
      if (output_placement_[idx] == DeviceType::kCPU) {
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": "
                  << reinterpret_cast<int8_t*>(result)[i];
        }
      } else {
        int64_t bytes = nelem * sizeof(int8_t);
        int* h_result = nullptr;
        if (nelem) {
          h_result = new int[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        for (int i = 0; i < nelem; ++i) {
          VLOG(2) << "  result #" << i << ": " << h_result[i];
        }
        actual_results_[idx] = std::shared_ptr<void>(
            h_result, [](void* p) { delete[] reinterpret_cast<int8_t*>(p); });
      }
    } else {
      return Internal("Error: unsupported output element type: ",
                      out_elem_types_[idx]);
    }
#else
    if (output_placement_[idx] != DeviceType::kCPU) {
      return Internal(
          "unexpected output placement, expect host tag for cpu only build");
    }
    if (out_elem_types_[idx] == tensorflow::DT_FLOAT) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<float*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_DOUBLE) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<double*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_HALF) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<half*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT32 ||
               out_elem_types_[idx] == tensorflow::DT_QINT32) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<int32_t*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT64) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<int64_t*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_BOOL) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<bool*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_UINT8 ||
               out_elem_types_[idx] == tensorflow::DT_QUINT8) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<uint8_t*>(result)[i];
      }
    } else if (out_elem_types_[idx] == tensorflow::DT_INT8 ||
               out_elem_types_[idx] == tensorflow::DT_QINT8) {
      for (int i = 0; i < nelem; ++i) {
        VLOG(2) << "  result #" << i << ": "
                << reinterpret_cast<int8_t*>(result)[i];
      }
    } else {
      return Internal("Error: unsupported output element type: ",
                      out_elem_types_[idx]);
    }
#endif
  }

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  for (int idx = 0; idx < num_inputs_; ++idx) {
    if (input_placement_[idx] != DeviceType::kCPU &&
        d_addr_vec[idx] != nullptr) {
      gpu_dealloc(d_addr_vec[idx]);
    }
  }
#endif

  return tsl::OkStatus();
}

}  //  namespace mlir_test
