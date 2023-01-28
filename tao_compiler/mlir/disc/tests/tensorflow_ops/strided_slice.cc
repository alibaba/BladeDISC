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
TEST(TFStridedSliceOpTest, StaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"3x4xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
}

// static shape test case with negative strides
TEST(TFStridedSliceOpTest, StaticShapeNegativeStrides2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_s_f32_2.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"3x4xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
}

// fully dynamic shape test case
TEST(TFStridedSliceOpTest, FullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"200x300xf32_X", "2xi32_h", "2xi32_h"},
      /*output_descriptors*/ {"f32_X", "f32_X"},
      /*input_vals*/ {{}, {-1, 0}, {-1, -1}}));
}

// fully dynamic shape test case
TEST(TFStridedSliceOpTest, FullyDynamicShapeNegativeStrides2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_d_f32_2.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"3x4xf32_X", "2xi32_h", "2xi32_h"},
      /*output_descriptors*/ {"f32_X", "f32_X"},
      /*input_vals*/ {{}, {0, 1}, {2, 0}}));
}

// fully dynamic shape 3d test case
TEST(TFStridedSliceOpTest, FullyDynamicShape3DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_d_3d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"200x300x3xf32_X", "2xi32_h", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 0}, {0, 1}}));
}

// partial dynamic shape test case
TEST(TFStridedSliceOpTest, PartialDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"200x300xf32_X", "2xi32_h", "2xi32_h"},
      /*output_descriptors*/ {"f32_X", "f32_X"},
      /*input_vals*/ {{}, {-1, 0}, {-1, -1}}));
}

// partial dynamic shape test case
TEST(TFStridedSliceOpTest, PartialDynamicShapeI32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_p_i32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"300xi32_X", "1xi32_h"},
      /*output_descriptors*/ {"i32_X"},
      /*input_vals*/ {{}, {2}}));
}

// partial dynamic shape test case
TEST(TFStridedSliceOpTest, PartialDynamicShapeI32Test2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_p_i32_2.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"300xi32_X"},
      /*output_descriptors*/ {"i32_X"},
      /*input_vals*/ {{}}));
}

// fully dynamic shape test case
TEST(TFStridedSliceOpTest, FullyDynamicShapeWithNewAxisAttr2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "strided_slice_with_newaxis_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"3x4xf32_X", "4xi32_h", "4xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 0, 0, 0}, {0, 0, 0, 0}}));
}

}  // namespace mlir_test
