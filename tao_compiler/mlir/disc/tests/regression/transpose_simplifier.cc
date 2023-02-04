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

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

TEST(TransposeSimplifierTest, ConvReluConvReluAddTest0) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "transpose_simplifier_test0.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"2x16x32x32xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"},
      /*input_vals*/ {}));
}

}  // namespace mlir_test