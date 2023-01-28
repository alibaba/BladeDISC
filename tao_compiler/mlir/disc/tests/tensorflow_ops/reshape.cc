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
TEST(TFReshapeOpTest, StaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x4xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape test case
TEST(TFReshapeOpTest, FullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"20x30x4xf32_X", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {600, 4}}));
}

// partial dynamic shape test case
TEST(TFReshapeOpTest, PartialShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"20x30x4xf32_X", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {600, 4}}));
}

// testing target shape contains minus one.
TEST(TFReshapeOpTest, MinusOneDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"20x30x4xf32_X", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {-1, 4}}));
}

// testing target shape contains minus one.
TEST(TFReshapeOpTest, MinusOneDynamicShape2DF32Test2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"20x30x4xf32_X", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {4, -1}}));
}

// TODO(disc): testing empty tensor reshape.

}  // namespace mlir_test
