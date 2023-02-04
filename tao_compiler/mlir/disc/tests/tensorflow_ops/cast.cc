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

// static shape test cast f32 to f16
TEST(TFCastOpTest, StaticShape2DF32ToF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "cast_s_f32tof16.mlir",
      // TODO(disc): cpu not support f16 codegen a.t.m.
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// fully dynamic shape test cast f32 to f16
TEST(TFCastOpTest, FullyDynamicShape2DF32ToF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "cast_d_f32tof16.mlir",
      // TODO(disc): cpu not support f16 codegen a.t.m.
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf32_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// partial dynamic shape test cast f32 to f16
TEST(TFCastOpTest, PartialShape2DF32ToF16) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "cast_p_f32tof16.mlir",
      // TODO(disc): cpu not support f16 codegen a.t.m.
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x100xf32_X"},
      /*output_descriptors*/ {"f16_X"}));
}

// test with provided data
TEST(TFCastOpTest, ProvidedDataShape2DF32ToF16) {
  EXPECT_TRUE(feature_test_main(
      // TODO(disc): cpu not support f16 codegen a.t.m.
      /*mlir_file_path*/ c_ft_path + "cast_d_f32tof16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x5xf32_X"},
      /*output_descriptors*/ {"f16_X"},
      /*input_vals*/ {{-2.4, -0.1, 0., 0.01, 3.2}}));
}

// static shape test cast f16 to f32
TEST(TFCastOpTest, StaticShape2DF16ToF32) {
  EXPECT_TRUE(feature_test_main(
      // TODO(disc): cpu not support f16 codegen a.t.m.
      /*mlir_file_path*/ c_ft_path + "cast_s_f16tof32.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf16_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// static shape test cast f32 to i32
TEST(TFCastOpTest, StaticShape2DF32ToI32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "cast_s_f32toi32.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xf32_X"},
      /*output_descriptors*/ {"i32_X"}));
}

// static shape test cast i32 to i1
TEST(TFCastOpTest, StaticShape2DI32ToI1) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "cast_s_i32toi1.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x100xi32_X"},
      /*output_descriptors*/ {"i1_X"}));
}

}  // namespace mlir_test
