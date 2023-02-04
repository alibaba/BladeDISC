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

const std::string c_ft_path = "mlir/disc/tests/tensorflow_ops/data/";

// static shape test case
TEST(TFFloorDivOpTest, StaticShape2DF32) {
  // Disable any fast-math setting for this UT.
  setenv("DISC_CUDA_FAST_MATH_LEVEL", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "floor_div_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X", "100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CUDA_FAST_MATH_LEVEL");
}

// fully dynamic shape test case
TEST(TFFloorDivOpTest, FullyDynamicShape2DF32) {
  // Disable any fast-math setting for this UT.
  setenv("DISC_CUDA_FAST_MATH_LEVEL", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "floor_div_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X", "100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CUDA_FAST_MATH_LEVEL");
}

// partial dynamic shape test case
TEST(TFFloorDivOpTest, PartialShape2DF32) {
  // Disable any fast-math setting for this UT.
  setenv("DISC_CUDA_FAST_MATH_LEVEL", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "floor_div_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x13xf32_X", "100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CUDA_FAST_MATH_LEVEL");
}

// implicit broadcast test case
TEST(TFFloorDivOpTest, ImplicitBroadcast2DF32) {
  // Disable any fast-math setting for this UT.
  setenv("DISC_CUDA_FAST_MATH_LEVEL", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "floor_div_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x100xf32_X", "17x1xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CUDA_FAST_MATH_LEVEL");
}

// test with provided data
TEST(TFFloorDivOpTest, ProvidedDataShape2DI64) {
  // Disable any fast-math setting for this UT.
  setenv("DISC_CUDA_FAST_MATH_LEVEL", "0", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "floor_div_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3xf32_X", "1x3xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{-2, 0, 23}, {3, 0, -22}}));
  unsetenv("DISC_CUDA_FAST_MATH_LEVEL");
}

}  // namespace mlir_test
