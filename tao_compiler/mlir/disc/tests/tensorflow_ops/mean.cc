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
    "tensorflow/compiler/mlir/disc/tests/tensorflow_ops/data/";

// dynamic shape 3D column reduction test case
TEST(TFMeanOpTest, ColReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_col_d_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D column reduction test case
TEST(TFMeanOpTest, ColReduceStaticShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_col_s_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D column reduction test case
TEST(TFMeanOpTest, ColReducePartialDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_col_p_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 3D row reduction test case
TEST(TFMeanOpTest, RowReduceFullyDynamicShape3DLargeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_row_d_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x12321xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D row reduction test case
TEST(TFMeanOpTest, RowReduceStaticShape3DSmallF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_row_s_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D row reduction test case
TEST(TFMeanOpTest, RowReducePartialDynamicShape3DSmallF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_row_p_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape 3D row reduction test case
TEST(TFMeanOpTest, MultidimReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "mean_multidim_d_f32.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

} // namespace mlir_test
