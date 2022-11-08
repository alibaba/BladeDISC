// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_calibrator.h"
#include "compiler/tensorrt/bridge/tensorrt_logger.h"

namespace torch {
namespace blade {
namespace tensorrt {

cudaError_t copyTensorToCuda(
    torch::Tensor& input,
    void* dst,
    size_t count,
    cudaMemcpyKind kind) {
  c10::ScalarType scalar_type = input.type().scalarType();
  cudaError_t err;
  switch (scalar_type) {
    case c10::ScalarType::Long: {
      int64_t* input_data = input.data_ptr<int64_t>();
      err = cudaMemcpy(dst, input_data, count * sizeof(int64_t), kind);
      break;
    }

    case c10::ScalarType::Int: {
      int32_t* input_data = input.data_ptr<int32_t>();
      err = cudaMemcpy(dst, input_data, count * sizeof(int32_t), kind);
      break;
    }
    case c10::ScalarType::Double: {
      double* input_data = input.data_ptr<double>();
      err = cudaMemcpy(dst, input_data, count * sizeof(double), kind);
      break;
    }
    case c10::ScalarType::Float: {
      float* input_data = input.data_ptr<float>();
      err = cudaMemcpy(dst, input_data, count * sizeof(float), kind);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported tensor data type.")
  }
  return err;
}

Int8EntropyCalibrator2Impl::Int8EntropyCalibrator2Impl(
    const CalibDataType& calib_data) {
  calib_data_ = calib_data;
  batch_num_ = calib_data_.size();
  input_num_ = calib_data_[0].size();
  batch_size_ = calib_data_[0][0].sizes()[0];
  // TODO: support batch data with different size
  auto first_inp = calib_data_[0];
  for (int i = 0; i < input_num_; i++) {
    input_count_.push_back(first_inp[i].numel());
    device_inputs_.push_back(nullptr);

    c10::ScalarType scalar_type = first_inp[i].type().scalarType();
    cudaError_t err;
    switch (scalar_type) {
      case c10::ScalarType::Long:
        err = cudaMalloc(&device_inputs_[i], input_count_[i] * sizeof(int64_t));
        break;
      case c10::ScalarType::Int:
        err = cudaMalloc(&device_inputs_[i], input_count_[i] * sizeof(int32_t));
        break;
      case c10::ScalarType::Double:
        err = cudaMalloc(&device_inputs_[i], input_count_[i] * sizeof(double));
        break;
      case c10::ScalarType::Float:
        err = cudaMalloc(&device_inputs_[i], input_count_[i] * sizeof(float));
        break;
      default:
        TORCH_CHECK(false, "Unsupported calibration data type.")
    }
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc failed: " << cudaGetErrorString(err);
    }
  }
}

bool Int8EntropyCalibrator2Impl::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings) {
  if (cur_batch_ >= batch_num_) {
    return false;
  }
  for (int i = 0; i < input_num_; i++) {
    auto input = calib_data_[cur_batch_][i];
    cudaError_t err = copyTensorToCuda(
        input, device_inputs_[i], input_count_[i], cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMemcpy failed: " << cudaGetErrorString(err);
      return false;
    }
    bindings[i] = device_inputs_[i];
  }
  cur_batch_++;
  return true;
}

Int8EntropyCalibrator2Impl::~Int8EntropyCalibrator2Impl() {
  for (int i = 0; i < input_num_; i++) {
    cudaError_t err = cudaFree(device_inputs_[i]);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(err);
    }
  }
}

} // namespace tensorrt
} // namespace blade
} // namespace torch
