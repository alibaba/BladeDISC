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

#include "torch_disc/csrc/disc_class.h"

namespace torch_disc {
namespace compiler {

static auto disc_class =
    torch::class_<DiscClass>("torch_disc", "DiscClass")
        .def("Run", [](const c10::intrusive_ptr<DiscClass>& self,
                       const torch::List<torch::Tensor>& inputs) {
          return self->Run(inputs);
        });

DiscClass::DiscClass(std::shared_ptr<DiscClassOption>& option) {
  ral_ctx_ = std::make_unique<torch::blade::RalContext>(
      option->executable_prog_bytes, option->constant_bytes,
      option->input_type_spec_str, option->output_type_spec_str,
      option->input_dev_str, option->output_dev_str);
  CHECK_NOTNULL(ral_ctx_);
}

torch::List<torch::Tensor> DiscClass::Run(
    const torch::List<torch::Tensor>& inputs) {
  auto ret = ral_ctx_->Forward(inputs);
  return ret;
}

}  //  namespace compiler
}  //  namespace torch_disc
