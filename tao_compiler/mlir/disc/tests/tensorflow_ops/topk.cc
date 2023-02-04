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

TEST(TFTopKOpTest, DynamicShape2DF32Small) {
  EXPECT_TRUE(
      feature_test_main(/*mlir_file_path*/ c_ft_path + "topk_d_f32.mlir",
                        /*backend_type*/ {BackendType::kCuda},
                        /*num_inputs*/ 2,
                        /*num_outputs*/ 2,
                        /*input_descriptors*/ {"2x16xf32_d", "i32_h"},
                        /*output_descriptors*/ {"f32_d", "i32_d"},
                        /*input_vals*/ {{}, {3}}));
}

TEST(TFTopKOpTest, DynamicShape2DF32Large) {
  EXPECT_TRUE(
      feature_test_main(/*mlir_file_path*/ c_ft_path + "topk_d_f32.mlir",
                        /*backend_type*/ {BackendType::kCuda},
                        /*num_inputs*/ 2,
                        /*num_outputs*/ 2,
                        /*input_descriptors*/ {"1x74541xf32_d", "i32_h"},
                        /*output_descriptors*/ {"f32_d", "i32_d"},
                        /*input_vals*/ {{}, {6}}));
}

TEST(TFTopKOpTest, StaticShape2DF32) {
  EXPECT_TRUE(
      feature_test_main(/*mlir_file_path*/ c_ft_path + "topk_s_f32.mlir",
                        /*backend_type*/ {BackendType::kCuda},
                        /*num_inputs*/ 2,
                        /*num_outputs*/ 2,
                        /*input_descriptors*/ {"2x16xf32_d", "i32_h"},
                        /*output_descriptors*/ {"f32_d", "i32_d"},
                        /*input_vals*/ {{}, {3}}));
}

TEST(TFTopKOpTest, PartialDynamicShape2DF32) {
  EXPECT_TRUE(
      feature_test_main(/*mlir_file_path*/ c_ft_path + "topk_p_f32.mlir",
                        /*backend_type*/ {BackendType::kCuda},
                        /*num_inputs*/ 2,
                        /*num_outputs*/ 2,
                        /*input_descriptors*/ {"2x16xf32_d", "i32_h"},
                        /*output_descriptors*/ {"f32_d", "i32_d"},
                        /*input_vals*/ {{}, {3}}));
}

TEST(TFTopKOpTest, DynamicShape2DI32) {
  EXPECT_TRUE(
      feature_test_main(/*mlir_file_path*/ c_ft_path + "topk_d_i32.mlir",
                        /*backend_type*/ {BackendType::kCuda},
                        /*num_inputs*/ 2,
                        /*num_outputs*/ 2,
                        /*input_descriptors*/ {"2x16xi32_d", "i32_h"},
                        /*output_descriptors*/ {"i32_d", "i32_d"},
                        /*input_vals*/ {{}, {3}}));
}

}  // namespace mlir_test
