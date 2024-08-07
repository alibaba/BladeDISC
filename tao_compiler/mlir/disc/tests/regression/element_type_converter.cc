/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

TEST(ElementTypeConverterTest, Gemm) {
  setenv("TAO_MLIR_ENABLE_AMP", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "element_type_converter_gemm.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"3x4xf32_X", "4x3xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
  unsetenv("TAO_MLIR_ENABLE_AMP");
}

TEST(ElementTypeConverterTest, ConvReluConvReluAddTest0) {
  setenv("TAO_MLIR_ENABLE_AMP", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "element_type_converter_conv.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x2x2x2xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
  unsetenv("TAO_MLIR_ENABLE_AMP");
}

}  // namespace mlir_test
