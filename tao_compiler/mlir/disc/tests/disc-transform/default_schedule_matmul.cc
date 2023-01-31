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
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING", "1", 1);
  return true;
}();

TEST(DynamicShapeMatmul, F32_24x24x64_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x64xf32_X", "64x24xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul, F32_24x768x768_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x768xf32_X", "768x768xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul, F32_24x3072x768_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x768xf32_X", "768x3072xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul, F32_24x768x3072_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x3072xf32_X", "3072x768xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul_nt, F32_304x1024x512_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_nt_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x512xf32_X", "1024x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul_tn, F32_304x1024x512_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_tn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"512x24xf32_X", "512x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(DynamicShapeMatmul_tt, F32_304x1024x512_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "default_schedule_matmul_tt_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"512x24xf32_X", "1024x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(StaticShapeMatmul, F32_24x24x64_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_s_24x24x64_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x64xf32_X", "64x24xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(StaticShapeMatmul, F32_24x64x24_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_s_24x64x24_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x24xf32_X", "24x64xf32_X"},
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
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_p_512x1024_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(PackedMatmul, F32_2x768_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_p_2x768_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x2xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(PackedMatmul, F32_768x2_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_p_768x2_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x768xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(PackedMatmul, F32_768x3072_Using_Default_Schedule) {
  EnvSetting setting = {{"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
                        {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
                        {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "default_schedule_matmul_nn_p_3072x768_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"24x3072xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

}  // namespace mlir_test
