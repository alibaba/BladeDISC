/* Copyright 2021 The BladeDISC Authors. All Rights Reserved.

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
#include <tensorflow/compiler/decoupling/mlir/disc/tests/mlir_test.h>
class DiscReply {
 public:
  explicit DiscReply(const std::string& input_pb, const std::string& tao_input_args): 
    args_(tao_input_args){ }

  bool Run() {
    // 1. parse the input tensors proto files into Tensor array
    if (!args_.Parsing()) {
      return false;
    }
    // 2. compile the input into binary
    if (!compilation.Run()) {
      return false;
    }

    // 3. run the compiled result with input Tensors 
    return RunExecutable();
  }

  bool RunExecutable() {

  }

 private:
  std::string tao_input_pb_;
  std::string tao_input_args_;
  ReplyArgs args_;
};

int main(int argc, char** argv) {
  std::string fn_ir("");
  std::string fn_inputs("");
  DiscReply r(fn_ir, fn_inputs);
  auto s = r.Run();
  if (!s.OK()) {
    std::err << s.Message() << std::endl;
  }
  return 0;
}