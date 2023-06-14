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

const std::string c_ft_path = "mlir/disc/tests/tensorflow_ops/data/";

TEST(TFSparseSegmentSumOpTest, DynamicShapeF32Dim2Input) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_segment_sum_d_f32_2d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_Xescriptors*/ {"4x4xf32_X", "6xi32_X", "6xi32_X"},
      /*output_Xescriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 1, 2, 0, 1, 3}, {0, 0, 0, 2, 2, 2}}));
}

TEST(TFSparseSegmentSumOpTest, PartialDynamicShapeF64Dim2Input) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_segment_sum_p_f64_2d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_Xescriptors*/ {"4x4xf64_X", "6xi32_X", "6xi32_X"},
      /*output_Xescriptors*/ {"f64_X"},
      /*input_vals*/ {{}, {0, 1, 2, 0, 1, 3}, {0, 0, 0, 2, 2, 2}}));
}

TEST(TFSparseSegmentSumOpTest, StaticShapeF32Dim2Input) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "sparse_segment_sum_s_f32_2d.mlir",
      /*backend_types*/ {kSupportedCPUBackendList},
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_Xescriptors*/ {"4x4xf32_X", "6xi32_X", "6xi32_X"},
      /*output_Xescriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {0, 1, 2, 0, 1, 3}, {0, 0, 0, 2, 2, 2}}));
}

}  // namespace mlir_test
