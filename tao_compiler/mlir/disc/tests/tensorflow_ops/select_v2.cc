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

// static shape + same shape test case
TEST(TFSelectV2OpTest, StaticShapeSameShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "select_v2_s_same_shape_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"10x11xi1_X", "10x11xf32_X", "10x11xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape + same shape test case
TEST(TFSelectV2OpTest, FullyDynamicShapeSameShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "select_v2_d_same_shape_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"10x11xi1_X", "10x11xf32_X", "10x11xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape + same shape test case
TEST(TFSelectV2OpTest, PartialShapeSameShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "select_v2_p_same_shape_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"10x11xi1_X", "10x11xf32_X", "10x11xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape + bcast shape test case
TEST(TFSelectV2OpTest, FullyDynamicShapeBcastShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "select_v2_d_bcast_shape_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4xi1_X", "3x4xf32_X", "8x3x4xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
