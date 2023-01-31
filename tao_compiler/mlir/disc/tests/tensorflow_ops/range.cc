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

// test with provided data to test i64
TEST(TFRangeOpTest, ProvidedDataShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "range_s_i64.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"i64_X", "i64_X", "i64_X"},
      /*output_descriptors*/ {"i64_X"},
      /*input_vals*/ {{8}, {11}, {2}}));
}

// test with provided data
TEST(TFRangeOpTest, ProvidedDataShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "range_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"f32_X", "f32_X", "f32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{8}, {11}, {2}}));
}

// test with provided data
TEST(TFRangeOpTest, DynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "range_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"f32_X", "f32_X", "f32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{8}, {15}, {6}}));
}

// test with provided data (reverse mode)
TEST(TFRangeOpTest, ReverseF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "range_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"f32_X", "f32_X", "f32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{32}, {-1}, {-1}}));
}

}  // namespace mlir_test
