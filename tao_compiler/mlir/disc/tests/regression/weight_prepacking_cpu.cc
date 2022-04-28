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

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/regression/data/";

TEST(WeightPrepackingTest, MatmulOnednnNotPacking) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "onednn", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "weight_prepacking_cpu_matmul.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

TEST(WeightPrepackingTest, MatmulOnednnPacking) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "onednn", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "weight_prepacking_cpu_matmul.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

TEST(WeightPrepackingTest, MatmulMKLNotPacking) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "mkl", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "weight_prepacking_cpu_matmul.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

TEST(WeightPrepackingTest, MatmulMKLPacking) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "mkl", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "weight_prepacking_cpu_matmul.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

}  // namespace mlir_test
