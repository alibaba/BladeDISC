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
TEST(EmptyTensor, FullyDynamicShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "empty_tensor.mlir",
      /*backend_types*/ kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"100x0xf32_X", "100x0xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// implicit broadcast test case
TEST(EmptyTensor, ImplicitBroadcast2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "empty_tensor.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 2,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"1x100xf32_X", "0x1xf32_X"},
      /*output_descriptors*/ {"f32_X"}));
}

// Corner case
// Empty reshape + matmul
TEST(EmptyTensor, EmptyReshapeMatmul) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "empty_tensor_reshape_matmul.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 3,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"0x2x3xf32_X", "6x11xf32_X", "2xi32_h"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{}, {}, {0, 6}}));
}

}  // namespace mlir_test
