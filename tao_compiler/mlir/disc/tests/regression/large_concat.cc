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

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/regression/data/";

TEST(LargeConcat, CPUTest) {
  std::vector<std::string> input_descriptors;
  for (int i = 0; i < 64; ++i) {
    input_descriptors.push_back("30x4xf32_X");
  }
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "large_concat_cpu.mlir",
      /*backend_types*/ {BackendType::kX86},
      /*num_inputs*/ input_descriptors.size(),
      /*num_outputs*/ 1,
      /*input_descriptors*/ input_descriptors,
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
