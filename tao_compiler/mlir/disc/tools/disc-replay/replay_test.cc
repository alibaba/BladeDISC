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

#include <cstdio>

#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/tar_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace replay {
namespace testing {

tensorflow::Tensor GetTestingTensor() {
  tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({10, 20}));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<float>()(a, b) = static_cast<float>(a * b);
    }
  }
  return t;
}

tensorflow::Status GetTestingRecordTar(const std::string& tar_fname) {
  auto env = tensorflow::Env::Default();
  std::string tmp_dir;
  env->LocalTempFilename(&tmp_dir);
  env->CreateDir(tmp_dir);

  tensorflow::tao::TaoCompilerInput input;
  tensorflow::TensorProto tensor_proto;
  GetTestingTensor().AsProtoTensorContent(&tensor_proto);

  // set argument kind
  auto arg = input.add_args();
  arg->set_kind_v2(tensorflow::tao::ArgumentKind::kConstant);

  // set tensor value
  std::string value_fn = "input_1.pb";
  std::string tao_input_pb_fn = "tao_compiler_input.pb";
  tensorflow::WriteBinaryProto(tensorflow::Env::Default(),
                               tensorflow::io::JoinPath(tmp_dir, value_fn),
                               tensor_proto);
  arg->set_value_proto_file(value_fn);

  tensorflow::WriteBinaryProto(
      tensorflow::Env::Default(),
      tensorflow::io::JoinPath(tmp_dir, tao_input_pb_fn), input);
  std::vector<std::string> srcs = {value_fn, tao_input_pb_fn};
  return CompressTar(srcs, tar_fname, tmp_dir);
}

std::string GetTempTarfileName() {
  auto env = tensorflow::Env::Default();
  std::string tmp_fname;
  env->LocalTempFilename(&tmp_fname);
  return tmp_fname + ".tar";
}

TEST(BladeDISCReplayTest, TestRecordArgs) {
  auto tar_fname = GetTempTarfileName();
  EXPECT_TRUE(GetTestingRecordTar(tar_fname).ok());

  ReplayRecord record;
  EXPECT_TRUE(record.InitFromTar(tar_fname).ok());
  auto tensors = record.Tensors();
  // auto placements = record.Placements();
  EXPECT_TRUE(tensors.size() == 1);

  // data value on the index 64(3 * 20 + 4) is equal to tensor[2][3]
  float expected_value = 12;
  EXPECT_EQ(tensors[0].flat<float>()(64), expected_value);
}

TEST(BladeDISCReplayTest, TestInterPreter) {
  // the test.tar is generate by quick start demo with enabling debug
  // mode(DISC_DEBUG=true)
  std::string tar_fname =
      "tensorflow/compiler/mlir/disc/tools/disc-replay/test_data/test.tar";
  DiscInterpreter disc;
  ReplayRecord record;
  CompiledResult result;
  // 1. Init replay record from tarball
  EXPECT_TRUE(record.InitFromTar(tar_fname).ok());
  // 2. compile the input program into executable program
  EXPECT_TRUE(disc.Compile(record.Program(), result).ok());
  // 3. run the executable with input data
  EXPECT_TRUE(disc.Run(result, record.Tensors(), record.Placements()).ok());
}

}  //  namespace testing
}  //  namespace replay
