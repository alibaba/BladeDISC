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

TEST(UintCastTest, UintCastTest) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "uint_cast.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{255, 128, 0, 63, 158, 33}}));
}

TEST(TFConstOpTest, ScalarUI8) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "uint_cast_const.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"4xui8_X"},
      /*output_descriptors*/ {"f32_X"}));
}

}  // namespace mlir_test
