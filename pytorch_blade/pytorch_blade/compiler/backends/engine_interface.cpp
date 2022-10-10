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

#include "pytorch_blade/compiler/backends/engine_interface.h"

#include <torch/script.h>
#include "common_utils/logging.h"

namespace torch {
namespace blade {
namespace backends {

namespace {
struct EngineCreatorRegistry {
  std::unordered_map<std::string, EngineCreatorRegister::EngineCreator> lut;

  EngineCreatorRegistry& RegisterBackend(
      const std::string& name,
      const EngineCreatorRegister::EngineCreator& creator) {
    auto found = lut.find(name);
    if (found != lut.end()) {
      LOG(FATAL) << "Overriding already registered EngineCreator for backend "
                 << name << ", unexpected behavior may occur.";
    }
    lut[name] = creator;
    return *this;
  }

  std::shared_ptr<EngineInterface> CreateEngine(const EngineState& state) {
    auto creator = lut.find(state.backend_name);
    if (creator == lut.end()) {
      LOG(WARNING) << "Can not found backend " << state.backend_name;
      return nullptr;
    }
    TORCH_CHECK(
        creator != lut.end(),
        "The EngineCreator for backend ",
        state.backend_name,
        " hasn't been registered.");
    return creator->second(state);
  }

  static EngineCreatorRegistry& GetRegistry() {
    static EngineCreatorRegistry registry;
    return registry;
  }
};
} // namespace

EngineState::SerialType EngineState::Serialize() const {
  std::vector<TensorType::SerialType> inputs_serialized;
  std::vector<TensorType::SerialType> outputs_serialized;
  inputs_serialized.reserve(inputs.size());
  outputs_serialized.reserve(outputs.size());

  for (auto& inp : inputs) {
    inputs_serialized.emplace_back(std::move(inp.Serialize()));
  }
  for (auto& out : outputs) {
    outputs_serialized.emplace_back(std::move(out.Serialize()));
  }
  return std::make_tuple(
      std::move(engine_bytes),
      std::move(model_proto),
      std::move(backend_name),
      std::move(inputs_serialized),
      std::move(outputs_serialized),
      std::move(extra_attrs));
}

EngineState EngineState::Deserialize(const SerialType& serialized) {
  EngineState state;
  state.engine_bytes = std::move(std::get<0>(serialized));
  state.model_proto = std::move(std::get<1>(serialized));
  state.backend_name = std::move(std::get<2>(serialized));
  auto& inputs_serialized = std::get<3>(serialized);
  auto& outputs_serialized = std::get<4>(serialized);

  state.inputs.reserve(inputs_serialized.size());
  state.outputs.reserve(outputs_serialized.size());
  for (auto& inp_s : inputs_serialized) {
    state.inputs.emplace_back(std::move(TensorType::Deserialize(inp_s)));
  }

  for (auto& out_s : outputs_serialized) {
    state.outputs.emplace_back(std::move(TensorType::Deserialize(out_s)));
  }
  state.extra_attrs = std::move(std::get<5>(serialized));
  return state;
}

const EngineCreatorRegister& EngineCreatorRegister::RegisterBackend(
    const std::string& name,
    const EngineCreator& creator) const {
  EngineCreatorRegistry::GetRegistry().RegisterBackend(name, creator);
  return *this;
}

std::shared_ptr<EngineInterface> EngineInterface::CreateEngine(
    const EngineState& state) {
  return EngineCreatorRegistry::GetRegistry().CreateEngine(state);
}

} // namespace backends
} // namespace blade
} // namespace torch
