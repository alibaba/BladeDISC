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

TEST(TFSparseFillEmptyRowsOpTest, DynamicShapeI64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_fill_empty_rows_d_i64.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 4,
      /*num_outputs*/ 4,
      /*input_Xescriptors*/ {"3x2xi64_X", "3xi64_X", "2xi64_X", "i64_X"},
      /*output_Xescriptors*/ {"i64_X", "i64_X", "i1_X", "i64_X"},
      /*input_vals*/ {{0, 0, 0, 1, 2, 0}, {4, 5, 6}, {4, 4}, {0}}));
}

TEST(TFSparseFillEmptyRowsOpTest, DynamicShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_fill_empty_rows_d_f32.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 4,
      /*num_outputs*/ 4,
      /*input_Xescriptors*/ {"6x2xi64_X", "6xf32_X", "2xi64_X", "f32_X"},
      /*output_Xescriptors*/ {"i64_X", "f32_X", "i1_X", "i64_X"},
      /*input_vals*/
      {{0, 0, 1, 0, 1, 3, 1, 4, 3, 2, 3, 3},
       {0.0, 10.0, 13.0, 14.0, 32.0, 33.0},
       {5, 6},
       {-1.0}}));
}

TEST(TFSparseFillEmptyRowsOpTest, PartialDynamicShapeF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_fill_empty_rows_p_f32.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 3,
      /*num_outputs*/ 4,
      /*input_Xescriptors*/ {"6x2xi64_X", "6xf32_X", "f32_X"},
      /*output_Xescriptors*/ {"i64_X", "f32_X", "i1_X", "i64_X"},
      /*input_vals*/
      {{0, 0, 1, 0, 1, 3, 1, 4, 3, 2, 3, 3},
       {0.0, 10.0, 13.0, 14.0, 32.0, 33.0},
       {-1.0}}));
}

}  // namespace mlir_test
