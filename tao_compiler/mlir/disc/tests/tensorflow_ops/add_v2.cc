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
TEST(TFAddV2OpTest, StaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "add_v2_s_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X", "100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape test case
TEST(TFAddV2OpTest, FullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "add_v2_d_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X", "100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFAddV2OpTest, PartialShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "add_v2_p_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x13xf32_X", "100x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast test case
TEST(TFAddV2OpTest, ImplicitBroadcast2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "add_v2_d_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x100xf32_X", "17x1xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
