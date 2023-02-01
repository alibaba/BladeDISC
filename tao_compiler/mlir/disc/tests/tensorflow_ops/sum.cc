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

// dynamic shape 2D column reduction test case
TEST(TFSumOpTest, ColReduceFullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_d_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 2D column reduction test case
TEST(TFSumOpTest, ColReduceFullyDynamicShape2DF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_d_2d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape 2D column reduction test case
TEST(TFSumOpTest, ColReduceFullyDynamicShape2DI8) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_d_2d_i8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xi8_X"},
      /*output_descriptors*/ {"i8_X"}));
}

// dynamic shape 3D column reduction test case
TEST(TFSumOpTest, ColReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_d_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 2D column reduction test case
TEST(TFSumOpTest, ColReduceStaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_s_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D column reduction test case
TEST(TFSumOpTest, ColReduceStaticShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_s_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 2D column reduction test case
TEST(TFSumOpTest, ColReducePartialDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_p_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D column reduction test case
TEST(TFSumOpTest, ColReducePartialDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_p_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 2D row reduction test case
TEST(TFSumOpTest, RowReduceFullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_d_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 2D column reduction test case
TEST(TFSumOpTest, ColReduceDynamicShape2DF64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_col_d_2d_f64.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf64_X"},
      /*output_descriptors*/ {"f64_X"}));
}

// dynamic shape 3D row reduction test case
TEST(TFSumOpTest, RowReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_d_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 3D row reduction test case
TEST(TFSumOpTest, RowReduceFullyDynamicShape3DF64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_d_3d_f64.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xf64_X"},
      /*output_descriptors*/ {"f64_X"}));
}

// static shape 2D row reduction test case
TEST(TFSumOpTest, RowReduceStaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 2D row reduction with i32 test case
TEST(TFSumOpTest, RowReduceStaticShape2DI32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_2d_i32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xi32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// static shape 2D row reduction with i64 test case
TEST(TFSumOpTest, RowReduceStaticShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_2d_i64.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xi64_X"},
      /*output_descriptors*/ {"i64_X"}));
}

// static shape 2D row reduction with f16 test case
TEST(TFSumOpTest, RowReduceStaticShape2DF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_2d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// static shape 3D row reduction test case
TEST(TFSumOpTest, RowReduceStaticShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D row reduction with i64 test case
TEST(TFSumOpTest, RowReduceStaticShape3DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_3d_i64.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xi64_X"},
      /*output_descriptors*/ {"i64_X"}));
}

// static shape 3D row reduction with f16 test case
TEST(TFSumOpTest, RowReduceStaticShape3DF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_s_3d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// partial dynamic shape 2D row reduction test case
TEST(TFSumOpTest, RowReducePartialDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_p_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D row reduction test case
TEST(TFSumOpTest, RowReducePartialDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sum_row_p_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x4096xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
