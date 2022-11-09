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

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include "NvInfer.h"

#include "pytorch_blade/compiler/backends/engine_interface.h"

namespace torch {
namespace blade {
namespace tensorrt {

using torch::blade::backends::CalibDataType;
using torch::blade::backends::PerInputCalibDataType;

class Int8EntropyCalibrator2Impl {
 public:
  Int8EntropyCalibrator2Impl(const CalibDataType& calib_data);

  int getBatchSize() const {
    return batch_size_;
  }

  bool getBatch(void* bindings[], const char* names[], int nbBindings);

  const void* readCalibrationCache(size_t& length) {
    return nullptr;
  }
  void writeCalibrationCache(const void* cache, size_t length) {}

 private:
  CalibDataType calib_data_;
  PerInputCalibDataType batch_data_;
  int batch_num_ = 0;
  int input_num_ = 0;
  int batch_size_ = 0;
  int cur_batch_ = 0;
};

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator2(const CalibDataType& calib_data)
      : calib_impl(calib_data) {}

  int getBatchSize() const noexcept override {
    return calib_impl.getBatchSize();
  }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
      override {
    return calib_impl.getBatch(bindings, names, nbBindings);
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    return calib_impl.readCalibrationCache(length);
  }

  void writeCalibrationCache(const void* cache, size_t length) noexcept
      override {
    calib_impl.writeCalibrationCache(cache, length);
  }

 private:
  Int8EntropyCalibrator2Impl calib_impl;
};

} // namespace tensorrt
} // namespace blade
} // namespace torch
