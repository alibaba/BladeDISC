#include "tensorrt_calibrator.h"
#include "compiler/tensorrt/bridge/tensorrt_logger.h"

#include <iostream>

namespace torch {
namespace blade {
namespace tensorrt {

Int8EntropyCalibrator2Impl::Int8EntropyCalibrator2Impl(
    const AllCalibDataType& grp_calib_data) {
  grp_calib_data_ = grp_calib_data;
  batch_num_ = grp_calib_data_.size();
  input_num_ = grp_calib_data_[0].size();
  batch_size_ = grp_calib_data_[0][0].sizes()[0];
  // TODO: support batch data with different size
  auto first_inp = grp_calib_data_[0];
  for (int i = 0; i < input_num_; i++) {
    input_count_.push_back(first_inp[i].numel());
    device_inputs_.push_back(nullptr);
    cudaError_t err =
        cudaMalloc(&device_inputs_[i], input_count_[i] * sizeof(float));
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
    auto input = grp_calib_data_[cur_batch_][i];
    auto input_data = input.data_ptr<float>();
    cudaError_t err = cudaMemcpy(
        device_inputs_[i],
        input_data,
        input_count_[i] * sizeof(float),
        cudaMemcpyHostToDevice);
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
