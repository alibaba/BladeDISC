// Copyright 2021 The BladeDISC Authors. All rights reserved.
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
#include <torch/script.h>

#include "common_utils/macros.h"
#include "compiler/mlir/runtime/ral_context.h"

namespace torch {
namespace blade {

class DiscEngineClass : public torch::CustomClassHolder {
 public:
  using AttrDict = c10::Dict<std::string, std::string>;
  using State = AttrDict;

  DISALLOW_COPY_AND_ASSIGN(DiscEngineClass);

  DiscEngineClass(State state);

  torch::List<torch::Tensor> Forward(const torch::List<torch::Tensor>& inputs);

  const std::string& GetOriginalSubgraph() {
    return original_subgraph_str_;
  }

  State Serialize();
  static c10::intrusive_ptr<DiscEngineClass> Deserialize(
      DiscEngineClass::State state);

  static State MakeState(
      const std::string& disc_debug_name,
      const std::string& ral_engine_bytes,
      const std::string& ral_const_bytes,
      const std::string& input_type_spec,
      const std::string& output_type_spec,
      const std::string& original_subgraph,
      const std::string& input_dev_str,
      const std::string& output_dev_str);

  torch::List<torch::Tensor> last_inputs();
  torch::List<torch::Tensor> last_outputs();

 private:
  std::string disc_debug_name_;
  std::string input_type_spec_;
  std::string output_type_spec_;
  std::string input_dev_str_;
  std::string output_dev_str_;
  std::string ral_engine_bytes_;
  std::string ral_const_bytes_;
  std::string original_subgraph_str_;
  torch::List<torch::Tensor> last_inputs_;
  torch::List<torch::Tensor> last_outputs_;

  // TODO: add fallback, refactor against TRTEngineClass
  std::unique_ptr<RalContext> ral_context_;
};

torch::TypePtr register_disc_engine(
    torch::jit::Module& module,
    const std::string& engine_debug_name,
    const std::string& ral_engine_bytes,
    const std::string& ral_const_bytes,
    const std::string& original_subgraph,
    const std::vector<torch::jit::Value*>& input_values,
    const std::vector<torch::jit::Value*>& output_values,
    const std::string& input_dev_str,
    const std::string& output_dev_str);
} // namespace blade
} // namespace torch
