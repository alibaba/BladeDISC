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

#include <fstream>
#include <mutex>
#include <tuple>

#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/compiler/backends/engine_interface.h"

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
class Module;
} // namespace jit
} // namespace torch

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
  at::List<at::Tensor> Execute(const at::List<at::Tensor>& inputs);
  void DumpAttrToFile(const std::string&, const std::string& dump_file) const;
  void DumpModelProto(const std::string& dump_file) const;
  std::string GetAttrString(const std::string&) const;
  std::vector<std::string> GetAttrKeys() const;

  SerialType Serialize();
  static c10::intrusive_ptr<EngineClass> Deserialize(
      EngineClass::SerialType serialized);

  static SerialType Serialize(
      EngineState state,
      std::string attr_debug_name,
      std::string fallback_module_bytes,
      std::string original_subgraph);

  at::List<at::Tensor> last_inputs();
  at::List<at::Tensor> last_outputs();

 private:
  torch::jit::Module GetFallback();
  at::List<at::Tensor> Fallback(const at::List<at::Tensor>& inputs);

  std::once_flag fallback_loaded_;
  std::string attr_debug_name_;
  AttrDictType attr_dict_;
  c10::intrusive_ptr<c10::ivalue::Object> fallback_module_;
  std::shared_ptr<EngineInterface> engine_;
  at::List<at::Tensor> last_inputs_;
  at::List<at::Tensor> last_outputs_;
  bool should_error_fallback_ = false;
};

c10::TypePtr register_engine(
    torch::jit::Module& module,
    const EngineState& engine_state,
    const std::string& attr_name,
    const std::string& fallback_module_bytes,
    const std::string& original_subgraph);

c10::IValue create_engine(
    const EngineState& engine_state,
    const std::string& attr_name,
    const std::string& fallback_module_bytes,
    const std::string& original_subgraph);

bool InitTorchBladeEngine();
} // namespace backends
} // namespace blade
} // namespace torch
