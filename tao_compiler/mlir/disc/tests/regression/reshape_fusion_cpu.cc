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

TEST(ReshapeFusionCPUTest, kLoopTest) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_fusion_cpu_kloop.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"12x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
}

TEST(ReshapeFusionCPUTest, kInputTest) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_fusion_cpu_kinput.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"12x13xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
}

TEST(ReshapeFusionCPUTest, kStitchTest) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_fusion_cpu_kstitch.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"12x13xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
}

TEST(ReshapeFusionCPUTest, kStitchTest2) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "reshape_fusion_cpu_kstitch_test2.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x20x560xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
}

}  // namespace mlir_test
