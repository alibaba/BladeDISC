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

#include "mlir/disc/tools/disc-replay/record.h"

#include "mlir/disc/tools/disc-replay/tar_helper.h"
#include "tensorflow/core/framework/tensor.h"
namespace replay {

std::string DeriveInputPlacement(
    const tensorflow::tao::ArgumentProto& arg_proto,
    const std::string& default_device) {
  if (arg_proto.kind_v2() == tensorflow::tao::ArgumentKind::kConstant) {
    return "const";
  } else if (arg_proto.kind_v2() ==
             tensorflow::tao::ArgumentKind::kFixedShaped) {
    return "cpu";
  } else if (arg_proto.kind_v2() == tensorflow::tao::ArgumentKind::kHostArgs) {
    return "cpu";
  } else {
    return default_device;
  }
}

tensorflow::Status ReadTensorFromPb(const std::string fname,
                                    tensorflow::Tensor* tensor) {
  TensorProto tensor_proto;
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(tensorflow::Env::Default(), fname, &tensor_proto));
  tensor->FromProto(tensor_proto);
  return tsl::OkStatus();
}

tensorflow::Status CheckTarCommand() {
  if (system("which tar > /dev/null 2>&1")) {
    return tensorflow::errors::Internal(
        "Can not find tar command tool, please install it before using this "
        "replay toolkit.");
  }
  return tsl::OkStatus();
}

tensorflow::Status ReplayRecord::Load() {
  TF_RETURN_IF_ERROR(CheckTarCommand());
  auto env = tensorflow::Env::Default();
  std::string out_dir;
  env->LocalTempFilename(&out_dir);
  env->CreateDir(out_dir);

  TF_RETURN_IF_ERROR(DeCompressTar(tar_fname_, out_dir));

  TF_CHECK_OK(env->FileExists(program_fname_));
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(tensorflow::Env::Default(), program_fname_, &program_));
  for (size_t i = 0; i < program_.args_size(); ++i) {
    auto placement = DeriveInputPlacement(program_.args(i), "gpu");
    tensorflow::Tensor t;
    std::string tensor_fname =
        tensorflow::io::JoinPath(out_dir, program_.args(i).value_proto_file());
    TF_CHECK_OK(env->FileExists(tensor_fname));
    TF_RETURN_IF_ERROR(ReadTensorFromPb(tensor_fname, &t));

    placements_.push_back(placement);
    tensors_.push_back(t);
  }
  return tsl::OkStatus();
}

}  //  namespace replay
