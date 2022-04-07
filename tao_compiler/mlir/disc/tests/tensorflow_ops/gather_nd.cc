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
TEST(TFGatherNdOpTest, StaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "gather_nd_s_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"7x8x90xf32_X", "2x3x2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 0, 0, 1, 1, 0, 2, 3, 3, 4, 1, 4}}));
}

// fully dynamic shape test case
TEST(TFGatherNdOpTest, FullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "gather_nd_d_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"7x8x90xf32_X", "2x3x2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 0, 0, 1, 1, 0, 2, 3, 3, 4, 1, 4}}));
}

// partial dynamic shape test case
TEST(TFGatherNdOpTest, PartialShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "gather_nd_p_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda, BackendType::kX86, BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"7x8x90xf32_X", "2x3x2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 0, 0, 1, 1, 0, 2, 3, 3, 4, 1, 4}}));
}

}  // namespace mlir_test
