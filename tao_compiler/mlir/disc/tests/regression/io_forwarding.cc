/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir/disc/tests/mlir_feature_test.h"
#include "mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

TEST(RalIOForwarding, GPUGPUTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_gpu_gpu.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf32_d", "3x4xf32_d"},
      /*output_descriptors*/ {"f32_d", "f32_d"}));
}

TEST(RalIOForwarding, GPUCPUTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_gpu_cpu.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf32_d", "3x4xf32_d"},
      /*output_descriptors*/ {"f32_h", "f32_h"}));
}

TEST(RalIOForwarding, GPUCPUF16Test) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_gpu_cpu_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf16_d", "3x4xf16_d"},
      /*output_descriptors*/ {"f16_h", "f16_h"}));
}

TEST(RalIOForwarding, CPUGPUTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_cpu_gpu.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf32_h", "3x4xf32_h"},
      /*output_descriptors*/ {"f32_d", "f32_d"}));
}

TEST(RalIOForwarding, CPUGPUF16Test) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_cpu_gpu_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf16_h", "3x4xf16_h"},
      /*output_descriptors*/ {"f16_d", "f16_d"}));
}

TEST(RalIOForwarding, CPUCPUTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "io_forwarding_cpu_cpu.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x3xf32_h", "3x4xf32_h"},
      /*output_descriptors*/ {"f32_h", "f32_h"}));
}

}  // namespace mlir_test
