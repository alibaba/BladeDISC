// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/disc/tests/mlir_feature_test.h"
#include "mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

// GELU f16.
TEST(EpilogueTest, EpilogueGELUF16) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  // compute-intensive fusion should be used along with stitch fusion.
  setenv("DISC_ENABLE_STITCH", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_fusion_gemm_gelu_f16.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x16x128x768xf16_X", "1x16x768x768xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

// GELU f32.
TEST(EpilogueTest, EpilogueGELUF32) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  setenv("DISC_ENABLE_STITCH", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_fusion_gemm_gelu_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x16x128x768xf32_X", "1x16x768x768xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

// The GEMM has multiple indirect consumers.
TEST(EpilogueTest, EpilogueMultiConsumers) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  // compute-intensive fusion should be used along with stitch fusion.
  setenv("DISC_ENABLE_STITCH", "true", 1);

  setenv("DISC_EXPECTED_KERNELS_IN_UT", "2", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "epilogue_fusion_gemm_multi_consumers.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"1x16x128x768xf16_X", "1x16x768x768xf16_X"},
      /*output_descriptors*/ {"f16_X", "f16_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");

  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

// Multiple GEMM fusions in the graph.
TEST(EpilogueTest, EpilogueMultiFusions) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  // compute-intensive fusion should be used along with stitch fusion.
  setenv("DISC_ENABLE_STITCH", "true", 1);

  setenv("DISC_EXPECTED_KERNELS_IN_UT", "3", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_fusion_gemm_multi_fusion.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/
      {"1x16x128x768xf16_X", "1x16x768x768xf16_X", "1x16x768x768xf16_X"},
      /*output_descriptors*/ {"f16_X", "f16_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");

  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

// GEMM + transpose.
TEST(EpilogueTest, EpilogueTransposeBMM0213) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  // compute-intensive fusion should be used along with stitch fusion.
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "epilogue_fusion_transpose_bmm0213.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x16x128x768xf16_X", "1x16x768x768xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

}  // namespace mlir_test