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

// static shape test case
TEST(TFEqualOpTest, StaticShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "equal_s_i64.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"111x1xi64_X", "111x1xi64_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// fully dynamic shape test case
TEST(TFEqualOpTest, FullyDynamicShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "equal_d_i64.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xi64_X", "100x100xi64_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// partial dynamic shape test case
TEST(TFEqualOpTest, PartialShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "equal_p_i64.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"13x12xi64_X", "13x12xi64_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// implicit broadcast test case
TEST(TFEqualOpTest, ImplicitBroadcast2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "equal_d_i64.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x100xi64_X", "17x1xi64_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// test with provided data
TEST(TFEqualOpTest, ProvidedDataShape2DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "equal_d_i64.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3xi64_X", "1x3xi64_X"},
      /*output_descriptors*/ {"i1_X"},
      /*input_vals*/ {{-2, 0, 23}, {3, 0, -22}}));
}

}  // namespace mlir_test
