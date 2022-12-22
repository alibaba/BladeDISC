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

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/disc-transform/data/";

static bool init_threads = []() {
  setenv("OMP_NUM_THREADS", "1", 1);
  return true;
}();

TEST(PackedMatmul, F32_304x1024x512) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "packed_matmul_nn_p_f32_large_schedule.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "packed_matmul_nn_p_512x1024_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(PackedMatmul, F32_304x1024x512_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "packed_matmul_nn_p_512x1024_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

}  // namespace mlir_test
