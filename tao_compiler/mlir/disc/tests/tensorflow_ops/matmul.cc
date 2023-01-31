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
TEST(TFMatMulOpTest, StaticShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFMatMulOpTest, StaticShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nt_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFMatMulOpTest, StaticShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tn_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X", "110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test case
TEST(TFMatMulOpTest, StaticShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tt_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X", "100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tn_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X", "110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tt_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X", "100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFMatMulOpTest, PartialDynamicShapeNNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "110x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFMatMulOpTest, PartialDynamicShapeNTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nt_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X", "100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFMatMulOpTest, PartialDynamicShapeTNF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tn_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x112xf32_X", "100x108xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case
TEST(TFMatMulOpTest, PartialDynamicShapeTTF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tt_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x101xf32_X", "100x100xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeNNF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"108x112xf16_X", "112x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeNTF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nt_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"108x112xf16_X", "100x112xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeTNF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tn_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"112x108xf16_X", "112x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeTTF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tt_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"112x108xf16_X", "100x112xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// dynamic shape test case
TEST(TFMatMulOpTest, DynamicShapeTTF64) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_tt_d_f64.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"112x108xf64_X", "100x112xf64_X"},
      /*output_descriptors*/ {"f64_X"}));
}

// const weight test case
TEST(TFMatMulOpTest, ConstWeightF32) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "onednn", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_const_weight_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x110xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

}  // namespace mlir_test
