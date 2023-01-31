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

TEST(SimpleTest, MatMulF32_11x13x12) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_schedule.mlir", false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"11x12xf32_X", "12x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

TEST(SimpleTest, MatMulF32_111x131x121) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_schedule.mlir", false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"111x121xf32_X", "121x131xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

TEST(SimpleTest, MatMulF32_304x1024x256) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule.mlir", false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_1024x1024x1024) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule.mlir", false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1024x1024xf32_X", "1024x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x1024x256_2) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_2.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_1024x1024x1024_2) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_2.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1024x1024xf32_X", "1024x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x256x256_3) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_3.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x256xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x512x256_3) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_3.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x512xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x1024x256_3) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_3.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x256xf32_X", "256x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x1024x512_3) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_3.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X", "512x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_1024x1024x1024_3) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_3.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1024x1024xf32_X", "1024x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x1024x512_4) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_4.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X", "512x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_1024x1024x1024_4) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_4.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1024x1024xf32_X", "1024x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_1026x1024x1024_4) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_4.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1026x1024xf32_X", "1024x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

TEST(SimpleTest, MatMulF32_304x1024x512_5) {
  EnvSetting setting = {
      {"DISC_TRANSFORM_SCHEDULE_FILE",
       {"kGEMM::" + c_ft_path + "matmul_nn_d_f32_large_schedule_5.mlir",
        false}},
      {"DISC_ENABLE_TRANSFORM_SCHEDULE", {"1", false}},
      {"DISC_ENABLE_SHAPE_CONSTRAINT_IR", {"1", false}},
      {"DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", {"0", false}}};
  EnvSettingContext ctx(setting);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "matmul_nn_d_f32.mlir",
      /*backend_types*/ {BackendType::kAArch64},
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"304x512xf32_X", "512x1024xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {},
      /*expected_output_vals*/ {},
      /*profiling*/ true));
}

}  // namespace mlir_test
