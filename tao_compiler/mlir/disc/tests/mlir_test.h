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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TAO_TESTS_MLIR_TEST_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TAO_TESTS_MLIR_TEST_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/session.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#endif

#if TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"
#elif GOOGLE_CUDA
#include "cuda.h"
#endif

namespace tensorflow {
class Status;
}  // namespace tensorflow

namespace tao {
namespace ral {
class OutputBufferWrapper;
class BaseContext;
}  // namespace ral
}  // namespace tao

namespace mlir_test {

using buffer_shape_t = std::vector<int64_t>;
using ::Eigen::half;
using tensorflow::DataType;

enum class DeviceType { kCPU, kGPU };

enum class BackendType { kCuda, kX86, kAArch64 };

static const std::vector<BackendType> kSupportedBackendList{
    BackendType::kCuda,
    BackendType::kX86,
    BackendType::kAArch64,
};

static const std::vector<BackendType> kSupportedCPUBackendList{
    BackendType::kX86,
    BackendType::kAArch64,
};

class MlirTest {
 public:
  explicit MlirTest(const std::string& mlir_file_path,
                    const std::string& tmp_dir, const std::string& test_name,
                    int num_inputs, int num_outputs,
                    const std::vector<buffer_shape_t>& input_shapes,
                    const std::vector<tensorflow::DataType>& input_elem_types,
                    const std::vector<DeviceType>& input_placement,
                    const std::vector<std::vector<float>>& input_vals,
                    const std::vector<tensorflow::DataType>& out_elem_types,
                    const std::vector<DeviceType>& output_placement,
                    const std::vector<tensorflow::Tensor>& expected_output_vals,
                    bool profiling = false, bool multi_cc_mode = false,
                    bool multi_cc_mode_dbg_ptx_only = false);

  tensorflow::Status Run();

 protected:
  int CallBinary(std::string program_path, std::vector<std::string> args);
  tensorflow::Status CompileMlirToBinary();
  virtual tensorflow::Status GenerateInputAndRun() = 0;
  tensorflow::Status LoadGraph(const std::string& graph_file_name);
  tensorflow::Status RunGoldenTF();
  tensorflow::Status CompareResults();
  template <class T>
  static bool IsAcceptableNear(T a, T b, double rel_err_limit = 1e-2,
                               double abs_err_limit = 1e-4);

  // env variables and intermediate file path
  std::string tf_opt_path_;
  std::string dhlo_compiler_main_path_;
  std::string tf_mlir_translate_path_;
  std::string compiled_so_file_;

  // properties of DUT pattern
  const std::string mlir_file_path_;
  const std::string tmp_dir_;
  const std::string test_name_;
  const int num_inputs_;
  const int num_outputs_;
  const std::vector<buffer_shape_t> input_shapes_;
  const std::vector<tensorflow::DataType> input_elem_types_;
  const std::vector<DeviceType> input_placement_;
  const std::vector<std::vector<float>> input_vals_;
  const std::vector<tensorflow::DataType> out_elem_types_;
  const std::vector<DeviceType> output_placement_;
  bool profiling_;
  bool multi_cc_mode_;
  bool multi_cc_mode_dbg_ptx_only_;
  void* tao_ral_func_ptr_;

  std::unique_ptr<tensorflow::Session> sess_;
  // stimulus in host memory
  std::vector<std::shared_ptr<void>> h_data_;
  // actual run results in host memory
  std::vector<std::shared_ptr<void>> actual_results_;

  // Stores the expected output of the test.
  std::vector<tensorflow::Tensor> expected_output_vals_;

  std::unordered_map<std::string, std::string> output_tensor_name_map_;
};

class MlirTestImpl : public MlirTest {
 public:
  explicit MlirTestImpl(
      const std::string& mlir_file_path, const std::string& tmp_dir,
      const std::string& test_name, int num_inputs, int num_outputs,
      const std::vector<buffer_shape_t>& input_shapes,
      const std::vector<tensorflow::DataType>& input_elem_types,
      const std::vector<DeviceType>& input_placement,
      const std::vector<std::vector<float>>& input_vals,
      const std::vector<tensorflow::DataType>& out_elem_types,
      const std::vector<DeviceType>& output_placement,
      const std::vector<tensorflow::Tensor>& expected_output_vals,
      bool profiling = false, bool multi_cc_mode = false,
      bool multi_cc_mode_dbg_ptx_only = false);

  ~MlirTestImpl();

 private:
  /*virtual*/ tensorflow::Status GenerateInputAndRun() override;
  std::vector<std::unique_ptr<tao::ral::OutputBufferWrapper>> output_buffers_;
  std::unique_ptr<tao::ral::BaseContext> context_;
};

buffer_shape_t ParseInputDescriptor(const std::string& s,
                                    const BackendType& backend,
                                    tensorflow::DataType* dtype,
                                    DeviceType* placement);

void ParseOutputDescriptor(const std::string& s, const BackendType& backend,
                           tensorflow::DataType* dtype, DeviceType* placement);

}  //  namespace mlir_test

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TAO_TESTS_MLIR_TEST_H_
