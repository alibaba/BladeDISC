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

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path = "tensorflow/compiler/mlir/disc/tests/pdll/data/";

TEST(SimpleTest, FusedAddMul) {
  EnvSetting setting = {
      {"DISC_TF_PDLL_FILES", {c_ft_path + "qgemm_i8_per_channel.pdll", false}},
      {"DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT", {"true", false}}};
  EnvSettingContext ctx(setting);
  std::vector<float> inputs(4 * 25, -0.6);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {4, 71});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 12.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "qgemm_i8_per_channel.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4x25xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));
}

}  // namespace mlir_test
