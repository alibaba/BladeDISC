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

const std::string c_ft_path = "mlir/disc/tests/disc-transform/data/";

static bool init_threads = []() {
  setenv("OMP_NUM_THREADS", "8", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "1", 1);
  return true;
}();

TEST(SimpleMTTest, MatMulF32_111x131x121_Thread_8) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_multithread_nn_d_f32_schedule.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_multithread_nn_d_f32.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"111x121xf32_X", "121x131xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

TEST(SimpleTest, MatMulF32_304x1024x256) {
  EnvSetting setting = {{"DISC_TRANSFORM_SCHEDULE_FILE",
                         {"kGEMM::" + c_ft_path +
                              "matmul_multithread_nn_d_f32_large_schedule.mlir",
                          false}},
                        {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_multithread_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
