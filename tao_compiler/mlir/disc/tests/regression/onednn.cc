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

static bool init_disc_math_mode = []() {
  setenv("DISC_CPU_MATH_KERNEL_MODE", "onednn", 1);
  return true;
}();

TEST(ONEDNNTest, GEMM) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "onednn_gemm.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"50x199xf32_X", "199x50xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(ONEDNNTest, BatchGEMM) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "onednn_batch_gemm.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x10x50x199xf32_X", "11x10x199x50xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

}  // namespace mlir_test
