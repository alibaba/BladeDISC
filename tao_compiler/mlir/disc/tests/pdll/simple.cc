/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

TEST(SimpleTest, FusedAddMul) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "simple_fused_add_mul.pdll", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "simple_fused_add_mul.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x12xf32_X", "11x12xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

TEST(SimpleTest, FusedAddMulMultiResults) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES",
       {c_ft_path + "simple_fused_add_mul_multi_results.pdll", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "simple_fused_add_mul_multi_results.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"11x12xf32_X", "11x12xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
}

}  // namespace mlir_test
