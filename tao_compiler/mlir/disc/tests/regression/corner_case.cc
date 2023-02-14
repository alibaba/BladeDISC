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

#if CUDA_VERSION >= 11100
// This case fails with PTXAS of version 11.0. Log here to know this error
// ahead in case a future version of ptxas fails for the same reason.
TEST(KCornerCaseTest, CornerCaseErrorBeforePTXASX11100) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "corner_case_error_before_ptxas11100.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"209x503xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}
#endif

TEST(KCornerCaseTest, FusionOrderMismatch) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "corner_case_fusion_order_mismatch.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 5,
      /*num_outputs*/ 1,
      /*input_descriptors*/
      {"i32_X", "1xi32_X", "f32_X", "280xf32_X", "1x66x560xf32_X"},
      /*output_descriptors*/ {"i32_X"},
      /*input_vals*/ {{0}, {66}, {17.8885441}, {}, {}}));
}

}  // namespace mlir_test