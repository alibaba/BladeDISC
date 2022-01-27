#include <cstdio>
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/record_args.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/tar_helper.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/path.h"


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
  auto tmp_dir = "/tmp/disc_testing";
  env->CreateDir(tmp_dir);
  tensorflow::tao::TaoCompilerInput input;
  tensorflow::TensorProto tensor_proto;
  GetTestingTensor().AsProtoTensorContent(&tensor_proto);

  // set argument kind
  auto arg = input.add_args();
  arg->set_kind_v2(tensorflow::tao::ArgumentKind::kConstant);

  // set tensor value
  std::string value_fn = "input_1.pb";
  tensorflow::WriteBinaryProto(tensorflow::Env::Default(), tensorflow::io::JoinPath(tmp_dir, value_fn), tensor_proto);
  arg->set_value_proto_file(value_fn);

  tensorflow::WriteBinaryProto(tensorflow::Env::Default(), tensorflow::io::JoinPath(tmp_dir, "tao_compiler_input.pb"), input);
  return CompressTarGz(tmp_dir, tar_fname);
}


TEST(BladeDISCReplayTest, TestRecordArgs) {
  std::string tar_fname = "/tmp/disc_testing.tar";
  EXPECT_TRUE(GetTestingRecordTar(tar_fname).ok());


  ReplayRecord record;
  EXPECT_TRUE(record.InitFromTarGz(tar_fname).ok());
  auto tensors = record.Tensors();
  auto placements = record.Placements();
  EXPECT_TRUE(tensors.size() == 1);
  
  // data value on the index 64(3 * 20 + 4) is equal to tensor[2][3]
  float expected_value = 12;
  EXPECT_EQ(tensors[0].flat<float>()(64), expected_value);
}

} //  namespace testing
} //  namespace replay
