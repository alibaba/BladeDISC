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

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

// LayerNorm
TEST(LayerNormTest, LayerNorm3DF32) {
  std::vector<float> input_val;
  for (int64_t i = 0; i < 1 * 128 * 768; i++) {
    input_val.push_back(0.5);
  }

  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "layer_norm.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x128x768xf32_X"},
      /*output_descriptors*/ {"f32_X"}, {input_val}));
}

// LayerNorm on GPU with stitch. It tests the kernel number.
TEST(LayerNormTest, LayerNormGPUStitchOnly3DF32) {
  std::vector<float> input_val;
  for (int64_t i = 0; i < 1 * 128 * 768; i++) {
    input_val.push_back(0.5);
  }
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "layer_norm.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x128x768xf32_X"},
      /*output_descriptors*/ {"f32_X"}, {input_val}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

}  // namespace mlir_test