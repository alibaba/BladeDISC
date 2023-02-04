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

// boolean
TEST(TFWhereOpTest, DynamicShapeBoolean1DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_d_bool_1d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"13xi1_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, StaticShapeBoolean2DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_s_bool_2d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi1_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, PartialDynamicShapeBoolean3DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_bool_3d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x2x3xi1_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, DynamicShapeBoolean6DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_d_bool_6d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"3x2x1x10x2x3xi1_X"},
      /*output_descriptors*/ {"i64_X"}));
}

// integer
TEST(TFWhereOpTest, PartialDynamicShapeI85DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_i8_5d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"23x6x11x2x3xi8_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, PartialDynamicShapeI323DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_i32_3d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x2x3xi32_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, PartialDynamicShapeI643DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_i64_3d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x2x3xi64_X"},
      /*output_descriptors*/ {"i64_X"}));
}

// float
// Although half type WhereCPUOp is registered.
// "No WhereOp available for float16/half type on"
// Thus skip f16 test
TEST(TFWhereOpTest, PartialDynamicShapeF643DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_f64_3d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x2x3xf64_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, PartialDynamicShapeF323DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_p_f32_3d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x2x3xf32_X"},
      /*output_descriptors*/ {"i64_X"}));
}

TEST(TFWhereOpTest, DynamicShapeF324DInput) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "where_d_f32_4d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"8x11x2x3xf32_X"},
      /*output_descriptors*/ {"i64_X"}));
}

}  // namespace mlir_test
