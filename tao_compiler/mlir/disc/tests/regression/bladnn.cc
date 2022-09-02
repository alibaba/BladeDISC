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

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/regression/data/";

static bool init_disc_bladnn_mode = []() {
  setenv("BLADE_GEMM_TUNE_JIT", "1", 1);
  setenv("BLADE_GEMM_VERBOSE", "1", 1);
  return true;
}();

TEST(BLADNNTest, GEMM) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bladnn_gemm.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"50x199xf16_X", "199x50xf16_X"},
      /*output_descriptors*/ {"f16_X"},
      /*input_vals*/ {}));
}

TEST(BLADNNTest, BatchGEMM) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bladnn_batch_gemm.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x10x50x199xf16_X", "11x10x199x50xf16_X"},
      /*output_descriptors*/ {"f16_X"},
      /*input_vals*/ {}));
}

TEST(BLADNNTest, Conv2D) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "bladnn_conv.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x64x56x56xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

}  // namespace mlir_test
