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

// NHWC
// static shape test case with NHWC/VALID/FP32
TEST(TFDepthwiseConv2dNativeOpTest, StaticShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "depthwise_conv2d_native_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x4x5x3xf32_X", "1x1x3x2xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// NHWC
// static shape test case with NHWC/VALID/FP32
TEST(TFDepthwiseConv2dNativeOpTest, StaticShape4DNHWCF32Test2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "depthwise_conv2d_native_s_f32_2.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x4x5x3xf32_X", "3x3x3x2xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// partial dynamic shape test case with NHWC/VALID/FP32
TEST(TFDepthwiseConv2dNativeOpTest, PartialShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "depthwise_conv2d_native_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x4x5x3xf32_X", "3x3x3x2xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape test case with NHWC/SAME/FP32
TEST(TFDepthwiseConv2dNativeOpTest, FullyDynamicShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "depthwise_conv2d_native_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x4x5x3xf32_X", "3x3x3x2xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// fully dynamic shape + const weight test case with NHWC/SAME/FP32
TEST(TFDepthwiseConv2dNativeOpTest, ConstWeightShape4DNHWCF32) {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "onednn", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "depthwise_conv2d_native_const_weight_d_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x32x32x6xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  unsetenv("DISC_CPU_MATH_KERNEL_MODE");
}

}  // namespace mlir_test
