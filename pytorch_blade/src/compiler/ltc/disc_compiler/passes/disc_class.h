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

#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>

#include "compiler/mlir/runtime/ral_context.h"

namespace torch_disc {
namespace compiler {

struct DiscClassOption {
  std::string executable_prog_bytes;
  std::string constant_bytes;
  std::string input_type_spec_str;
  std::string output_type_spec_str;
  std::string input_dev_str;
  std::string output_dev_str;
};

class DiscClass : public torch::CustomClassHolder {
 public:
  DiscClass(std::shared_ptr<DiscClassOption>& option);

  static std::shared_ptr<DiscClassOption> MakeOption() {
    return std::make_shared<DiscClassOption>();
  }
  // Run executes Disc executable program with input tensors
  torch::List<torch::Tensor> Run(const torch::List<torch::Tensor>& inputs);

 private:
  std::shared_ptr<DiscClassOption> option_;
  std::unique_ptr<torch::blade::RalContext> ral_ctx_;
};

} //  namespace compiler
} //  namespace torch_disc