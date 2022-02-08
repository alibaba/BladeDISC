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

#ifndef DISC_REPLAY_RECORD_ARGS_H_
#define DISC_REPLAY_RECORD_ARGS_H_

#include "tensorflow/compiler/decoupling/tao_compiler_input.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/subprocess.h"

namespace replay {
using tensorflow::TensorProto;

class ReplayRecord {
 public:
  ReplayRecord(){};

  // Initialize ReplayRecord from a record file with tar.gz foramt.
  tensorflow::Status InitFromTar(const std::string& fname);

  tensorflow::tao::TaoCompilerInput& Program() { return input_; };
  std::vector<tensorflow::Tensor> Tensors() { return tensors_; };
  std::vector<std::string> Placements() { return placements_; };

 private:
  std::vector<tensorflow::Tensor> tensors_;
  std::vector<std::string> placements_;
  tensorflow::tao::TaoCompilerInput input_;
};

}  //  namespace replay

#endif  // DISC_REPLAY_RECORD_ARGS_H_