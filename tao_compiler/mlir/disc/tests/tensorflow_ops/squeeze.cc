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

// dynamic shape test with dim_size 1
TEST(TFSqueezeOpTest, DynamicShapeTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "squeeze_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x3xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test without dim_size 1
TEST(TFSqueezeOpTest, StaticShapeTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "squeeze_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x3xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test without dim_size 1
TEST(TFSqueezeOpTest, PartialDynamicShapeTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "squeeze_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x3xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test with dim_size -1
TEST(TFSqueezeOpTest, DynamicShapeTest2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "squeeze_d_f32_2.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x1xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test with dim_size -rank
TEST(TFSqueezeOpTest, DynamicShapeTest3) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "squeeze_d_f32_3.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x2xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
