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
#include <torch/script.h>

#include <fstream>
#include <mutex>
#include <tuple>

#include "common_utils/logging.h"
#include "common_utils/macros.h"
#include "compiler/backends/engine_interface.h"

namespace torch {
namespace blade {
namespace backends {

class EngineClass : public torch::CustomClassHolder {
 public:
  // The State is used by [de]serialization.
  // Each change to the State data structure,
  // will cause serialization version to promote.
  // To avoid changing the fields, please
  // add new debug fields to the debug attr dict
  using AttrDictType = c10::Dict<std::string, std::string>;
  using SerialType = std::tuple<
      EngineInterface::State::SerialType,
      AttrDictType // for extend if need
      >;

  DISALLOW_COPY_AND_ASSIGN(EngineClass);

  EngineClass(SerialType serialized);
  torch::List<torch::Tensor> Execute(const torch::List<torch::Tensor>& inputs);
  void DumpAttrToFile(const std::string&, const std::string& dump_file) const;
  std::string GetAttrString(const std::string&) const;
  std::vector<std::string> GetAttrKeys() const;

  SerialType Serialize();
  static c10::intrusive_ptr<EngineClass> Deserialize(
      EngineClass::SerialType serialized);

  static SerialType Serialize(
      EngineInterface::State state,
      std::string attr_debug_name,
      std::string fallback_module_bytes,
      std::string original_subgraph);

  torch::List<torch::Tensor> last_inputs();
  torch::List<torch::Tensor> last_outputs();

 private:
  torch::jit::Module GetFallback();
  torch::List<torch::Tensor> Fallback(const torch::List<torch::Tensor>& inputs);

  std::once_flag fallback_loaded_;
  std::string attr_debug_name_;
  AttrDictType attr_dict_;
  torch::jit::Module fallback_module_;
  std::shared_ptr<EngineInterface> engine_;
  torch::List<torch::Tensor> last_inputs_;
  torch::List<torch::Tensor> last_outputs_;
};

torch::TypePtr register_engine(
    torch::jit::Module& module,
    const EngineInterface::State& engine_state,
    const std::string& attr_name,
    const std::string& fallback_module_bytes,
    const std::string& original_subgraph);

} // namespace backends
} // namespace blade
} // namespace torch
