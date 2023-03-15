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

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

TEST(I8AddTest, I8AddTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_add_d_i8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi8_X", "2x1xi8_X"},
      /*output_descriptors*/ {"i8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(I8SubTest, I8SubTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_sub_d_i8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi8_X", "2x1xi8_X"},
      /*output_descriptors*/ {"i8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(I8MulTest, I8MulTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_mul_d_i8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi8_X", "2x1xi8_X"},
      /*output_descriptors*/ {"i8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(I8DivTest, I8DivTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_div_d_i8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi8_X", "2x1xi8_X"},
      /*output_descriptors*/ {"i8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(UI8AddTest, UI8AddTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_add_d_ui8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X", "2x1xui8_X"},
      /*output_descriptors*/ {"ui8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(UI8SubTest, UI8SubTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_sub_d_ui8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X", "2x1xui8_X"},
      /*output_descriptors*/ {"ui8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(UI8MulTest, UI8MulTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_mul_d_ui8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X", "2x1xui8_X"},
      /*output_descriptors*/ {"ui8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(UI8DivTest, UI8DivTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_div_d_ui8.mlir",
      // TODO(disc): FIXME: `Integer division by zero` error on AArch64
      /*backend_types*/
      {BackendType::kX86, /*BackendType::kAArch64,*/ BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X", "2x1xui8_X"},
      /*output_descriptors*/ {"ui8_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(I32AddTest, I32AddTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_add_d_i32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xi32_X", "2x1xi32_X"},
      /*output_descriptors*/ {"i32_X"},
      /*input_vals*/ {{-1, 127, 0, 63, -127, 33}, {1, -1}}));
}

TEST(IntTest, UI8ConvertTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "int_arithmetic_convert_d_ui8.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X"},
      /*output_descriptors*/ {"i1_X"},
      /*input_vals*/ {{1, 127, 0, 63, 22, 33}}));
}

}  // namespace mlir_test
