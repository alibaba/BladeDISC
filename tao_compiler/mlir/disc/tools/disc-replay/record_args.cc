#include "tensorflow/compiler/mlir/disc/tools/disc-replay/record_args.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/tar_helper.h"
#include "tensorflow/core/framework/tensor.h"
namespace replay {

std::string DeriveInputPlacement(const tensorflow::tao::ArgumentProto& arg_proto, const std::string& default_device) {
  if (arg_proto.kind_v2() == tensorflow::tao::ArgumentKind::kConstant) {
    return "const";
  } else if (arg_proto.kind_v2() == tensorflow::tao::ArgumentKind::kFixedShaped) {
    return "cpu";
  } else if (arg_proto.kind_v2() == tensorflow::tao::ArgumentKind::kHostArgs) {
    return "cpu";
  } else {
    return default_device;
  }
  return "";
}

tensorflow::Status ReadTensorFromPb(const std::string fname, tensorflow::Tensor* tensor) {
  TensorProto tensor_proto;
  TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(), fname, &tensor_proto));
  tensor->FromProto(tensor_proto);
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayRecord::InitFromTarGz(const std::string& fname) {
  auto env = tensorflow::Env::Default();
  std::shared_ptr<ReplayRecord> record;
  std::string out_dir = "/tmp";
  std::string basename = fname.substr(fname.find_last_of('/')+1);
  basename = basename.substr(0, basename.find_last_of('.'));

  TF_RETURN_IF_ERROR(DeCompressTar(fname, out_dir));

  std::string tao_input_fname = tensorflow::io::JoinPath(out_dir, basename, "tao_compiler_input.pb");
  TF_CHECK_OK(env->FileExists(tao_input_fname));
  TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(), tao_input_fname, &input_));
  for (size_t i = 0; i < input_.args_size(); ++i) {
    auto placement = DeriveInputPlacement(input_.args(i), "gpu");
    tensorflow::Tensor t;
    std::string tensor_fname = tensorflow::io::JoinPath(out_dir, basename, input_.args(i).value_proto_file());
    TF_CHECK_OK(env->FileExists(tensor_fname));
    TF_RETURN_IF_ERROR(ReadTensorFromPb(tensor_fname, &t));

    placements_.push_back(placement);
    tensors_.push_back(t);
  }
  return tensorflow::Status::OK();
}


} //  namespace replay
