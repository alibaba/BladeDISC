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

// fully dynamic shape test case with NHWC/SAME/FP32
TEST(TFConv2DBackpropFilterOpTest, FullDynamicShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "conv2d_backprop_filter_d_f32.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/
      {"4xi32_X", "100x28x28x1xf32_X", "100x26x26x32xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{3, 3, 1, 32}, {}, {}}));
}

// fully dynamic shape test case with NHWC/SAME/FP32
TEST(TFConv2DBackpropFilterOpTest, FullDynamicShape4DNHWCF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "conv2d_backprop_filter_d_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/
      {"4xi32_X", "100x28x28x1xf16_X", "100x26x26x32xf16_X"},
      /*output_descriptors*/ {"f16_X"},
      /*input_vals*/ {{3, 3, 1, 32}, {}, {}}));
}

// partial dynamic shape test case with NHWC/SAME/FP32
TEST(TFConv2DBackpropFilterOpTest, PartialShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "conv2d_backprop_filter_p_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/
      {"4xi32_X", "100x28x28x1xf32_X", "100x26x26x32xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{3, 3, 1, 32}, {}, {}}));
}

// static dynamic shape test case with NHWC/SAME/FP32
TEST(TFConv2DBackpropFilterOpTest, StaticShape4DNHWCF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "conv2d_backprop_filter_s_f32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x28x28x1xf32_X", "100x26x26x32xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}
}  // namespace mlir_test
