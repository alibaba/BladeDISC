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

Int8EntropyCalibrator2Impl::Int8EntropyCalibrator2Impl(
    const CalibDataType& calib_data) {
  calib_data_ = calib_data;
  batch_num_ = calib_data_.size();
  input_num_ = calib_data_[0].size();
  // TODO: support batch data with different size
  batch_size_ = calib_data_[0][0].sizes()[0];
  batch_data_.resize(input_num_);
}

bool Int8EntropyCalibrator2Impl::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings) {
  if (cur_batch_ >= batch_num_) {
    return false;
  }
  for (int i = 0; i < input_num_; i++) {
    // To save cuda memory, we lazily move calibration data
    // to gpu here.
    auto input = calib_data_[cur_batch_][i].cuda().contiguous();
    // Hold it so that the input tensor will not be destroyed
    // during calibration.
    batch_data_[i] = input;
    bindings[i] = input.data_ptr();
  }
  cur_batch_++;
  return true;
}

} // namespace tensorrt
} // namespace blade
} // namespace torch
