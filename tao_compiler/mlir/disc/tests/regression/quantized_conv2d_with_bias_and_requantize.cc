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

TEST(QuantizedConv2DWithBiasAndRequantizeOpTest, I8I8I8_NO_BIAS) {
  std::vector<float> inputs(1 * 2 * 2 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 2, 2, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_with_bias_and_requantize_p_i8_i8_i8_no_bias.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x2x2x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(QuantizedConv2DWithBiasAndRequantizeOpTest, I8I8I8_NO_BIAS_QINT32) {
  std::vector<float> inputs(1 * 2 * 2 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 2, 2, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_with_bias_and_requantize_p_i8_i8_i8_no_bias_qint32."
          "mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x2x2x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

}  // namespace mlir_test
