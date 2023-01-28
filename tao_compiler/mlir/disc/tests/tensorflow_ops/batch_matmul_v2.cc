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

// TODO(disc): re-enable after we support general cases of dot_general op.
// Currently tf2mhlo converter using tf2xla bridge to convert tf.MatMulV2Op
// with implicit broadcast to mhlo dialect. This will generates dot_general
// op with unusual format.

// // static shape test case
// TEST(TFBatchMatMulV2OpTest, StaticShapeNNF32) {
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_s_f32.mlir",
//       /*backend_types*/ {BackendType::kCuda, BackendType::kX86,
//       BackendType::kAArch64},
//       /*num_inputs*/ 2,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"1x3x100x110xf32_X", "2x1x110x100xf32_X"},
//       /*output_descriptors*/ {"f32_X"}));
// }

// partial dynamic shape test case
TEST(TFBatchMatMulV2OpTest, PartialDynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x100x110xf32_X", "3x1x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulV2OpTest, DynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulV2OpTest, DynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x100x110xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulV2OpTest, DynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_tn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFBatchMatMulV2OpTest, DynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_tt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x110x100xf32_X", "2x3x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x100x110xf32_X", "2x1x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x100x110xf32_X", "2x1x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_tn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x110x100xf32_X", "2x1x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_tt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x110x100xf32_X", "2x1x100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeNNF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x108x112xf16_X", "2x1x112x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// implicit broadcast dynamic shape test case
TEST(TFBatchMatMulV2OpTest, IBDynamicShapeNNF64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_d_f64.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x3x108x112xf64_X", "2x1x112x100xf64_X"},
      /*output_descriptors*/ {"f64_X"}));
}

// dynamic shape test case with m = 1
TEST(TFBatchMatMulV2OpTest, DynamicShapeM1F32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "batch_matmul_v2_nn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x1x110xf32_X", "2x3x110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
