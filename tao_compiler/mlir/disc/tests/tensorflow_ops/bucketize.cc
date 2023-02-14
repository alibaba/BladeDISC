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

// f32 static shape test case
TEST(TFBucketizeOpTest, StaticShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_s_f32.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x10xf32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// f32 dynamic shape test case
TEST(TFBucketizeOpTest, DynamicShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_d_f32.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4x20xf32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// f32 partial dynamic shape test case
TEST(TFBucketizeOpTest, PartialDynamicShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_p_f32.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"6x10xf32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// f64 static shape test case
TEST(TFBucketizeOpTest, StaticShapeF64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_s_f64.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x10xf64_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// i32 dynamic shape test case
TEST(TFBucketizeOpTest, DynamicShapeI32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_d_i32.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"8x8xi32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// i64 partial dynamic shape test case
TEST(TFBucketizeOpTest, PartialDynamicShapeI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bucketize_p_i64.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x10xi64_X"},
      /*output_descriptors*/ {"i32_X"}));
}

}  // namespace mlir_test
