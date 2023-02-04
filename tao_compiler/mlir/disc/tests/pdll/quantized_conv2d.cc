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

const std::string c_ft_path = "mlir/disc/tests/pdll/data/";

TEST(TFQuantziedConv2d, STATIC_SHAPE_NHWC_I8_PER_CHANNEL_1) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  EnvSettingContext ctx(setting);
  std::vector<float> inputs(1 * 2 * 2 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 2, 2, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nhwc_i8_per_channel_1.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x2x2x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NHWC_I8_PER_CHANNEL_2) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 8 * 8 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 6, 6, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 108.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nhwc_i8_per_channel_2.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x8x8x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NHWC_I8_PER_CHANNEL_3) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 8 * 8 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 2, 2, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 108.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nhwc_i8_per_channel_3.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x8x8x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NHWC_I8_PER_CHANNEL_4) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 8 * 8 * 25, -0.0);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 4, 4, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 0.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nhwc_i8_per_channel_4.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x8x8x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NHWC_I8_PER_CHANNEL_5) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  // input: NHWC
  //   [0,0,2,0,4, 0]
  //   [6,0,8,0,10,0]
  // kernel
  //   [0,1]
  //   [0,1]
  // stride = 2
  // output
  //   [0,0,0]
  std::vector<float> inputs(1 * 6 * 1 * 2);
  for (unsigned i = 0; i < 12; ++i) inputs[i] = (float)(i);
  for (unsigned i = 1; i < 12; i += 2) inputs[i] = 0.0;
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 3, 1, 1});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 0.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nhwc_i8_per_channel_5.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x5x5x3xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NCHW_I8_PER_CHANNEL_1) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  EnvSettingContext ctx(setting);
  std::vector<float> inputs(1 * 25 * 2 * 2, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 8, 2, 2});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nchw_i8_per_channel_1.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x25x2x2xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NCHW_I8_PER_CHANNEL_2) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 25 * 8 * 8, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 8, 6, 6});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 108.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nchw_i8_per_channel_2.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x25x8x8xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NCHW_I8_PER_CHANNEL_3) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 25 * 8 * 8, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 8, 2, 2});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 108.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nchw_i8_per_channel_3.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x25x8x8xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, STATIC_SHAPE_NCHW_I8_PER_CHANNEL_4) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 25 * 8 * 8, 0.0);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 8, 4, 4});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 0.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_s_nchw_i8_per_channel_4.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x25x8x8xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

TEST(TFQuantziedConv2d, PARTIAL_SHAPE_NCHW_I8_PER_CHANNE) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "quantized_conv2d_i8.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", "false"}}};
  std::vector<float> inputs(16 * 8 * 8 * 25, 0.0);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {16, 4, 4, 8});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 0.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "quantized_conv2d_p_nhwc_i8_per_channel.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"16x8x8x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

// Comment qconv2d int8 GPU test since it only supports NVIDIA Ampere
// architecture currently. Will add it back when ut running on machine whose GPU
// architecture >= Ampere

// TEST(TFQuantziedConv2d, PARTIAL_DYNAMIC_SHAPE_NHWC_I8_PER_TENSOR) {
//   std::vector<float> inputs(1 * 56 * 56 * 16, 1);
//   tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {1, 56, 56, 16});
//   auto datas = output.flat<float>();
//   for (int i = 0; i < output.NumElements(); ++i) datas(i) = 16;
//   EXPECT_TRUE(feature_test_main(
//       /*mlir_file_path*/ c_ft_path +
//           "quantized_conv2d_p_nhwc_i8_per_tensor.mlir",
//       /*backend_types*/ {BackendType::kCuda},
//       /*num_inputs*/ 1,
//       /*num_outputs*/ 1,
//       /*input_descriptors*/ {"1x56x56x16xf32_X"},
//       /*output_descriptors*/ {"f32_X"},
//       /*input_vals*/ {inputs},
//       /*expect_output_vals*/ {output}));
// }

}  // namespace mlir_test
