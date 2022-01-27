#include "record_args.h"
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
  std::shared_ptr<ReplayRecord> record;
  std::string folder;

  TF_RETURN_IF_ERROR(DeCompressTarGz(fname, &folder));

  std::string tao_input_fname = tensorflow::io::JoinPath("tao_compiler_input.pb");
  TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(), tao_input_fname, &input_));
  for (size_t i = 0; i < input_.args_size(); ++i) {
    auto placement = DeriveInputPlacement(input_.args(i), "gpu");
    tensorflow::Tensor t;
    TF_RETURN_IF_ERROR(ReadTensorFromPb(input_.args(i).value_proto_file(), &t));

    placements_.push_back(placement);
    tensors_.push_back(t);
  }
  return tensorflow::Status::OK();
}


} //  namespace replay
