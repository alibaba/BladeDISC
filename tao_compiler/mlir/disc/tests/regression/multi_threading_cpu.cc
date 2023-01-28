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

static bool init_threads = []() {
  setenv("OMP_NUM_THREADS", "3", 1);
  return true;
}();

TEST(MultiThreadingTest, 2DTest0) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_2d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"50x1xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 2DTest1) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_2d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"20x300xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 2DTest2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_2d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"60x100xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 3DTest1) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_3d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"5x1x1xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 3DTest2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_3d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x2xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 3DTest3) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_3d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x50xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 3DTest4) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_3d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3x50xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 4DTest1) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_4d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x2x2x2xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 4DTest2) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_4d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x1x1xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 4DTest3) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_4d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x1x1x100xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 4DTest4) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_4d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"5x2x1x100xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

TEST(MultiThreadingTest, 4DTest5) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "multi_threading_cpu_4d.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x2x1x100xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {}));
}

}  // namespace mlir_test
