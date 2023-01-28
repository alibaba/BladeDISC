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

const std::string c_ft_path = "mlir/disc/tests/tensorflow_ops/data/";

TEST(TFQuantziedMatMul, STATIC_SHAPE_I8_PER_CHANNEL) {
  std::vector<float> inputs(4 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {4, 71});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "quantized_matmul_s_i8_per_channel.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedMatMul, PARTIAL_DYNAMIC_SHAPE_NHWC_I8_PER_CHANNEL) {
  std::vector<float> inputs(4 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {4, 71});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "quantized_matmul_p_i8_per_channel.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

// Comment qgemm int8 GPU test since it only works when blade_gemm
// is on

// TEST(TFQuantziedMatMul, PARTIAL_DYNAMIC_SHAPE_NHWC_I8_PER_CHANNEL) {
//   std::vector<float> inputs(32 * 64, 2.0);
//   tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {32, 128});
//   auto datas = output.flat<float>();
//   for (int i = 0; i < output.NumElements(); ++i) datas(i) = 128.0;
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path + "quantized_matmul_p_i8_per_tensor.mlir",
//       /*backend_types*/ {BackendType::kCuda},
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"32x64xf32_X"},
//       /*output_descriptors*/ {"f32_X"},
//       /*input_vals*/ {inputs},
//       /*expect_output_vals*/ {output}));
// }

}  // namespace mlir_test
