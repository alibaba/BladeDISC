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
TEST(TFBatchMatMulOpTest, StaticShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nn_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFBatchMatMulOpTest, StaticShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nt_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFBatchMatMulOpTest, StaticShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tn_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFBatchMatMulOpTest, StaticShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tt_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFBatchMatMulOpTest, PartialDynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nn_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x100x110xf32_X", "1x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFBatchMatMulOpTest, PartialDynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nt_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x100x110xf32_X", "2x1x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFBatchMatMulOpTest, PartialDynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tn_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x100x112xf32_X", "1x3x100x108xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFBatchMatMulOpTest, PartialDynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tt_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x100x101xf32_X", "2x1x100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeNNF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nn_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x108x112xf16_X", "2x3x112x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeNTF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_nt_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x108x112xf16_X", "2x3x100x112xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeTNF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tn_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x112x108xf16_X", "2x3x112x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulOpTest, DynamicShapeTTF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_tt_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x112x108xf16_X", "2x3x100x112xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

}  // namespace mlir_test
