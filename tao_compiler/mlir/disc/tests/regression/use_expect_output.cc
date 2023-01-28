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

TEST(UseExpectOutput, BasicTest) {
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {2, 3});
  auto datas = output.flat<float>();
  datas(0) = 255.0f;
  datas(1) = 128.0f;
  datas(2) = 0.0f;
  datas(3) = 63.0f;
  datas(4) = 158.0f;
  datas(5) = 33.0f;

  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "use_expect_output.mlir",
      /*backend_types*/
      kSupportedBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"2x3xui8_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {{255, 128, 0, 63, 158, 33}},
      /*expect_output_vals*/ {output}));
}

}  // namespace mlir_test
