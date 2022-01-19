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

#include "compiler/mlir/runtime/disc_engine_class.h"

#include "common_utils/logging.h"
#include "common_utils/utils.h"

namespace torch {
namespace blade {
namespace {
void SetValueIfExists(
    const c10::Dict<std::string, std::string>& attr_dict,
    const std::string& key,
    std::string& value) {
  auto found = attr_dict.find(key);
  if (found != attr_dict.end()) {
    value = found->value();
  }
}
} // namespace

const char* kDiscDbgName = "disc_debug_name";
const char* kRalEngineBytes = "ral_engine_bytes";
const char* kRalConstBytes = "ral_const_bytes";
const char* kInputTypeSpec = "input_type_spec";
const char* kOutputTypeSpec = "output_type_spec";
const char* kInputDevStr = "input_dev_str";
const char* kOutputDevStr = "output_dev_str";
const char* kOriginalSubG = "original_subgraph";

DiscEngineClass::DiscEngineClass(State state) {
  auto& attr_dict = state;
  SetValueIfExists(attr_dict, kDiscDbgName, disc_debug_name_);
  SetValueIfExists(attr_dict, kRalEngineBytes, ral_engine_bytes_);
  SetValueIfExists(attr_dict, kRalConstBytes, ral_const_bytes_);
  SetValueIfExists(attr_dict, kInputTypeSpec, input_type_spec_);
  SetValueIfExists(attr_dict, kOutputTypeSpec, output_type_spec_);
  SetValueIfExists(attr_dict, kOriginalSubG, original_subgraph_str_);
  SetValueIfExists(attr_dict, kInputDevStr, input_dev_str_);
  SetValueIfExists(attr_dict, kOutputDevStr, output_dev_str_);
  ral_context_ = std::make_unique<RalContext>(
      ral_engine_bytes_,
      ral_const_bytes_,
      input_type_spec_,
      output_type_spec_,
      input_dev_str_,
      output_dev_str_);
  CHECK_NOTNULL(ral_context_);
}

torch::List<torch::Tensor> DiscEngineClass::last_inputs() {
  TORCH_CHECK(
      GetRecordClusterIOFlag(), "Only avaliable with RecordClusterIOFlag set");
  return last_inputs_;
}
torch::List<torch::Tensor> DiscEngineClass::last_outputs() {
  TORCH_CHECK(
      GetRecordClusterIOFlag(), "Only avaliable with RecordClusterIOFlag set");
  return last_outputs_;
}

torch::List<torch::Tensor> DiscEngineClass::Forward(
    const torch::List<torch::Tensor>& inputs) {
  if (GetRecordClusterIOFlag()) {
    // Note:
    // This is for accuracy debug & testing purpose, not recommend
    // to use at real runtime. Also, we won't support multi-threads.
    last_inputs_ = inputs;
  }

  auto record_inputs = std::vector<torch::IValue>{inputs.begin(), inputs.end()};
  RECORD_FUNCTION(disc_debug_name_, record_inputs);

  auto outputs = ral_context_->Forward(inputs);
  if (GetRecordClusterIOFlag()) {
    // Note:
    // This is for accuracy debug & testing purpose, not recommend
    // to use at real runtime. Also, we won't support multi-threads.
    last_outputs_ = outputs;
  }
  return outputs;
}

DiscEngineClass::State DiscEngineClass::Serialize() {
  return DiscEngineClass::MakeState(
      disc_debug_name_,
      ral_engine_bytes_,
      ral_const_bytes_,
      input_type_spec_,
      output_type_spec_,
      original_subgraph_str_,
      input_dev_str_,
      output_dev_str_);
}

DiscEngineClass::State DiscEngineClass::MakeState(
    const std::string& disc_debug_name,
    const std::string& ral_engine_bytes,
    const std::string& ral_const_bytes,
    const std::string& input_type_spec,
    const std::string& output_type_spec,
    const std::string& original_subgraph,
    const std::string& input_dev_str,
    const std::string& output_dev_str) {
  AttrDict attr_dict;
  attr_dict.insert(kDiscDbgName, disc_debug_name);
  attr_dict.insert(kRalEngineBytes, ral_engine_bytes);
  attr_dict.insert(kRalConstBytes, ral_const_bytes);
  attr_dict.insert(kInputTypeSpec, input_type_spec);
  attr_dict.insert(kOutputTypeSpec, output_type_spec);
  attr_dict.insert(kOriginalSubG, original_subgraph);
  attr_dict.insert(kInputDevStr, input_dev_str);
  attr_dict.insert(kOutputDevStr, output_dev_str);
  return attr_dict;
}

c10::intrusive_ptr<DiscEngineClass> DiscEngineClass::Deserialize(
    DiscEngineClass::State state) {
  return c10::make_intrusive<DiscEngineClass>(std::move(state));
}

static auto torch_blade_disc_class =
    torch::class_<DiscEngineClass>("torch_blade", "TaoEngine")
        .def(torch::init<DiscEngineClass::State>())
        .def(
            "forward",
            [](const c10::intrusive_ptr<DiscEngineClass>& self,
               const torch::List<torch::Tensor>& inputs) {
              return self->Forward(inputs);
            })
        .def("last_inputs", &DiscEngineClass::last_inputs)
        .def("last_outputs", &DiscEngineClass::last_outputs)
        .def(kOriginalSubG, &DiscEngineClass::GetOriginalSubgraph)
        .def_pickle(
            [](const c10::intrusive_ptr<DiscEngineClass>& self)
                -> DiscEngineClass::State { return self->Serialize(); },
            [](DiscEngineClass::State state)
                -> c10::intrusive_ptr<DiscEngineClass> {
              return DiscEngineClass::Deserialize(state);
            });

torch::TypePtr register_disc_engine(
    torch::jit::Module& module,
    const std::string& engine_debug_name,
    const std::string& ral_engine_bytes,
    const std::string& ral_const_bytes,
    const std::string& original_subgraph,
    const std::vector<const torch::jit::Value*>& input_values,
    const std::vector<const torch::jit::Value*>& output_values,
    const std::string& input_dev_str,
    const std::string& output_dev_str) {
  const auto& input_type_spec = ShapeTypeSpec::GetShapeTypeSpec(input_values);
  const auto& output_type_spec = ShapeTypeSpec::GetShapeTypeSpec(output_values);
  DiscEngineClass::State state = DiscEngineClass::MakeState(
      engine_debug_name,
      ral_engine_bytes,
      ral_const_bytes,
      input_type_spec.Serialize(),
      output_type_spec.Serialize(),
      original_subgraph,
      input_dev_str,
      output_dev_str);
  auto custom_class_obj =
      torch::make_custom_class<DiscEngineClass>(std::move(state));
  module.register_attribute(
      engine_debug_name, custom_class_obj.type(), custom_class_obj);
  return custom_class_obj.type();
}
} // namespace blade
} // namespace torch
