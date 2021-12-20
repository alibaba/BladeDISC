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
TEST(TFAnyOpTest, ColReduceNotKeepDimStaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "any_s_i1.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// fully dynamic shape test case
TEST(TFAnyOpTest, ReduceToScalarKeepDimFullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "any_d_i1.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X"},
      /*output_descriptors*/ {"i1_X"}));
}

// partial dynamic shape test case
TEST(TFAnyOpTest, ReduceToScalarKeepDimPartialShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "any_p_i1.mlir",
      /*backend_types*/ {BackendType::kCuda, BackendType::kX86},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X"},
      /*output_descriptors*/ {"i1_X"}));
}

}  // namespace mlir_test
