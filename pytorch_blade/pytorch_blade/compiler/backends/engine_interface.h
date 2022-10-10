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

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include "pytorch_blade/compiler/backends/backend_input_outputs.h"

namespace torch {
namespace blade {
namespace backends {

typedef std::vector<at::Tensor> PerInputCalibDataType;
typedef std::vector<PerInputCalibDataType> CalibDataType;

struct EngineState {
  typedef c10::Dict<std::string, std::string> AttrType;
  typedef torch::blade::backends::TensorInfo TensorType;

  TORCH_BLADE_BACKENDS_DEFINE_FIELD(engine_bytes, std::string);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(model_proto, std::string);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(backend_name, std::string);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(inputs, std::vector<TensorType>);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(outputs, std::vector<TensorType>);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(extra_attrs, AttrType);

  // As a medium for passing values from python to c++ and
  // will not be serialized

  TORCH_BLADE_BACKENDS_DEFINE_FIELD(calib_data, CalibDataType);

  using SerialType = std::tuple<
      std::string,
      std::string,
      std::string,
      std::vector<TensorType::SerialType>,
      std::vector<TensorType::SerialType>,
      AttrType>;

  SerialType Serialize() const;
  static EngineState Deserialize(const SerialType& serialized);
};

class EngineInterface {
 public:
  // The State is used by [de]serialization.
  // Each change to the State data structure,
  // will cause serialization version to promote.
  // To avoid changing the fields, please
  // add new debug fields to the debug attr dict
  using State = EngineState;
  virtual const State& GetState() const = 0;

  static const char* GetBackendName() {
    return "Engine";
  }

  virtual at::List<at::Tensor> Execute(const at::List<at::Tensor>& inputs) = 0;

  virtual bool ShouldFallback(const at::List<at::Tensor>& inputs) {
    return false;
  }

  static std::shared_ptr<EngineInterface> CreateEngine(const State&);
};

struct EngineCreatorRegister {
  typedef std::function<std::shared_ptr<EngineInterface>(const EngineState&)>
      EngineCreator;

  const EngineCreatorRegister& RegisterBackend(
      const std::string&,
      const EngineCreator&) const;
};

} // namespace backends
} // namespace blade
} // namespace torch
