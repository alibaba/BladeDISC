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
TEST(TFH2DOpTest, StaticShape1DI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "inline_h2d_s_i64.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1xi64_h", "1xi64_h", "1xi64_h"},
      /*output_descriptors*/ {"i64_d"}));
}

// kstitch style h2d fusion
TEST(TFH2DOpTest, kStitchStyleFusion) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "inline_h2d_d_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"31x30xf32_d"},
      /*output_descriptors*/ {"f32_d"}));
}

}  // namespace mlir_test
