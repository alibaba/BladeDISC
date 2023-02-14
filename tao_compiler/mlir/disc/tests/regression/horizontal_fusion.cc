/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

// fully dynamic shape test case
TEST(HorizontalTensor, BasicTest) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_ENABLE_HORIZONTAL_FUSION", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "horizontal_fusion.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"100x32xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_HORIZONTAL_FUSION");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
}

}  // namespace mlir_test
