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

// Test RandomUniform Op U(0,1) has : Mean(X) ~ E(X) = 0.5
TEST(TaoMlirFeatureTest, TF2XLA_RandomUniformTest1) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "random_uniform_d_mean.mlir",
      /*backend_type*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2xi32_h", "f32_d", "f32_d"},
      /*output_descriptors*/ {"i1_d"},
      /*input_vals*/ {{30, 400}, {0.5}, {0.01}}));
}

// Test RandomUniform Op U(0,1) has : Mean((X-E(X))^2) ~ E((X-E(X))^2) = 1/12
TEST(TaoMlirFeatureTest, TF2XLA_RandomUniformTest2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "random_uniform_d_var.mlir",
      /*backend_type*/ {BackendType::kCuda},
      /*num_inputs*/ 4,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2xi32_h", "f32_d", "f32_d", "f32_d"},
      /*output_descriptors*/ {"i1_d"},
      /*input_vals*/ {{300, 400}, {0.5}, {1. / 12}, {0.01}}));
}

// Test RandomUniform Op has :
//  - if two RandomUniform ops have same seed & seed2, then the results should
//  also be same
//  - E(XY) = 0.25 if X ~ U(0, 1) and Y ~ U(0, 1)
TEST(TaoMlirFeatureTest, TF2XLA_RandomUniformTest3) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "random_uniform_d_same_seed.mlir",
      /*backend_type*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2xi32_h", "f32_d", "f32_d"},
      /*output_descriptors*/ {"i1_d", "f32_d"},
      /*input_vals*/ {{300, 400}, {0.5}, {0.01}}));
}

// Test for static shape
TEST(TaoMlirFeatureTest, TF2XLA_RandomUniformTest4) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "random_uniform_s_same_seed.mlir",
      /*backend_type*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2xi32_h", "f32_d", "f32_d"},
      /*output_descriptors*/ {"i1_d", "f32_d"},
      /*input_vals*/ {{300, 400}, {0.5}, {0.01}}));
}

}  // namespace mlir_test
