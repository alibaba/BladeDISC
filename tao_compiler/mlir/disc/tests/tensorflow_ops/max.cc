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

// dynamic shape 3D column reduction test case
TEST(TFMaxOpTest, ColReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_col_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D column reduction test case
TEST(TFMaxOpTest, ColReduceStaticShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_col_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D column reduction test case
TEST(TFMaxOpTest, ColReducePartialDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_col_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape 3D row reduction test case
TEST(TFMaxOpTest, RowReduceFullyDynamicShape3DLargeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_row_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x12321xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 3D row reduction test case
TEST(TFMaxOpTest, RowReduceStaticShape3DSmallF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_row_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape 2D row reduction test case
TEST(TFMaxOpTest, RowReduceStaticShape2DSmallF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_row_s_2d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11000x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape 3D row reduction test case
TEST(TFMaxOpTest, RowReducePartialDynamicShape3DSmallF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_row_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape 3D row reduction test case
TEST(TFMaxOpTest, MultidimReduceFullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "max_multidim_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100x123xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
