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

#include "engine_class.h"

#include <torch/script.h>
#include "common_utils/logging.h"
#include "common_utils/utils.h"
#include "ltc/disc_compiler/replay.h"
#include "sys/stat.h"

namespace torch {
namespace blade {
namespace backends {

const char* kDebugName = "attr_debug_name";
const char* kOrigSubG = "module_graph";
const char* kFallbackModule = "fallback_module";

EngineClass::EngineClass(SerialType serialized) {
  const auto& engine_state =
      EngineState::Deserialize(std::move(std::get<0>(serialized)));
  engine_ = EngineInterface::CreateEngine(engine_state);
  TORCH_CHECK(engine_, "Create Engine failed!");

  attr_dict_ = std::move(std::get<1>(serialized));
  attr_debug_name_ = std::move(GetAttrString(kDebugName));
}

at::List<at::Tensor> EngineClass::Fallback(const at::List<at::Tensor>& inputs) {
  auto fallback = GetFallback();
  std::vector<IValue> f_inputs(inputs.begin(), inputs.end());

  auto ret = fallback.forward(f_inputs);
  TORCH_CHECK(
      ret.isTensorList(),
      "Only List[Tensor] is supported for outputs, please report a bug");
  return ret.toTensorList();
}

at::List<at::Tensor> EngineClass::Execute(const at::List<at::Tensor>& inputs) {
  if (GetRecordClusterIOFlag()) {
    // Note:
    // This is for accuracy debug & testing purpose, not recommend
    // to use at real runtime. Also, we won't support multi-threads.
    last_inputs_ = inputs;
  }

  if (torch_disc::compiler::IsEnableClusterReplayRecord()) {
    const auto& dump_path = "/tmp/replay_cluster_" + attr_debug_name_;
    TORCH_CHECK(
        !mkdir(dump_path.c_str(), 0755), "unable to create dir: " + dump_path);

    std::vector<c10::IValue> ivalues;
    for (auto input : inputs)
      ivalues.emplace_back(input);

    torch_disc::compiler::DumpIValues(ivalues, dump_path);

    auto graph_fname = dump_path + "/graph.pt";
    auto module = GetFallback();
    const auto method_name =
        torch::QualifiedName(*module.type()->name(), "forward");
    auto func =
        GetFallback()._ivalue()->compilation_unit()->find_function(method_name);
    auto graph = torch::jit::tryToGraphFunction(*func)->graph();

    VLOG(0) << "replay toolkit dump cluster on: " << dump_path
            << " , the sub-graph: \n"
            << graph->toString();
    GetFallback().save(graph_fname);
  }

  auto record_inputs = std::vector<torch::IValue>{inputs.begin(), inputs.end()};
  TORCH_BLADE_RECORD_FUNCTION(attr_debug_name_, record_inputs);

  at::List<at::Tensor> outputs;
  {
    auto module = GetFallback();
    const auto method_name =
        torch::QualifiedName(*module.type()->name(), "forward");
    auto func =
        GetFallback()._ivalue()->compilation_unit()->find_function(method_name);
    auto graph = torch::jit::tryToGraphFunction(*func)->graph();
    torch_disc::compiler::FoldOutputs(graph);
    VLOG(0) << "fallback graph: " << graph->toString();
  }
  outputs = Fallback(inputs);
  // do inference
  /**
  if (engine_->ShouldFallback(inputs)) {
    outputs = Fallback(inputs);
  } else {
    try {
      outputs = engine_->Execute(inputs);
    } catch (const std::runtime_error& error) {
      const auto& enable_fallback =
          env::ReadBoolFromEnvVar("TORCH_BLADE_ENABLE_RUNTIME_FALLBACK", true);
      if (enable_fallback) {
        outputs = Fallback(inputs);
      } else {
        throw error;
      }
    }
  }
  **/

  if (GetRecordClusterIOFlag()) {
    // Note:
    // This is for accuracy debug & testing purpose, not recommend
    // to use at real runtime. Also, we won't support multi-threads.
    last_outputs_ = outputs;
  }
  return outputs;
}

torch::jit::Module EngineClass::GetFallback() {
  std::call_once(fallback_loaded_, [&]() {
    const auto& fallback_module_bytes = GetAttrString(kFallbackModule);
    TORCH_CHECK(
        !fallback_module_bytes.empty(), "Fallback haven't been implemented");
    std::stringstream istream(fallback_module_bytes);
    fallback_module_ = ::torch::jit::load(istream)._ivalue();
  });
  return fallback_module_;
}

void EngineClass::DumpAttrToFile(
    const std::string& attr,
    const std::string& dump_file) const {
  std::ofstream writer(dump_file);
  writer << GetAttrString(attr);
  writer.close();
}

void EngineClass::DumpModelProto(const std::string& dump_file) const {
  std::ofstream writer(dump_file);
  writer << engine_->GetState().model_proto;
  writer.close();
}

std::string EngineClass::GetAttrString(const std::string& attr) const {
  if (attr_dict_.contains(attr)) {
    return attr_dict_.at(attr);
  } else {
    return std::string("");
  }
}

std::vector<std::string> EngineClass::GetAttrKeys() const {
  std::vector<std::string> keys;
  keys.reserve(attr_dict_.size());
  std::transform(
      attr_dict_.begin(),
      attr_dict_.end(),
      std::back_inserter(keys),
      [](auto& kv) { return kv.key(); });
  return keys;
}

at::List<at::Tensor> EngineClass::last_inputs() {
  TORCH_CHECK(
      GetRecordClusterIOFlag(), "Only avaliable with RecordClusterIOFlag set");
  return last_inputs_;
}
at::List<at::Tensor> EngineClass::last_outputs() {
  TORCH_CHECK(
      GetRecordClusterIOFlag(), "Only avaliable with RecordClusterIOFlag set");
  return last_outputs_;
}

EngineClass::SerialType EngineClass::Serialize(
    EngineState state,
    std::string attr_debug_name,
    std::string fallback_module_bytes,
    std::string original_subgraph) {
  AttrDictType attr_dict;
  attr_dict.insert(kDebugName, std::move(attr_debug_name));
  attr_dict.insert(kOrigSubG, std::move(original_subgraph));
  attr_dict.insert(kFallbackModule, std::move(fallback_module_bytes));
  return std::make_tuple(std::move(state.Serialize()), std::move(attr_dict));
}

EngineClass::SerialType EngineClass::Serialize() {
  return std::make_tuple(
      std::move(engine_->GetState().Serialize()), attr_dict_);
}

c10::intrusive_ptr<EngineClass> EngineClass::Deserialize(
    SerialType serialized) {
  return c10::make_intrusive<EngineClass>(std::move(serialized));
}

bool InitTorchBladeEngine() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    // Notice a few things:
    // - We pass the class to be registered as a template parameter to
    //   `torch::class_`. In this instance, we've passed the
    //   specialization of the MyStackClass class ``MyStackClass<std::string>``.
    //   In general, you cannot register a non-specialized template
    //   class. For non-templated classes, you can just pass the
    //   class name directly as the template parameter.
    // - The arguments passed to the constructor make up the "qualified name"
    //   of the class. In this case, the registered class will appear in
    //   Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
    //   the first argument the "namespace" and the second argument the
    //   actual class name.
    auto torch_blade_engine_class =
        // TODO(GTY): Use inheritance or template to do version control
        torch::class_<EngineClass>("torch_blade", "Engine")
            // The following line registers the contructor of our MyStackClass
            // class that takes a single `std::vector<std::string>` argument,
            // i.e. it exposes the C++ method `MyStackClass(std::vector<T>
            // init)`. Currently, we do not support registering overloaded
            // constructors, so for now you can only `def()` one instance of
            // `torch::init`.
            .def(torch::init<EngineClass::SerialType>())
            // The next line registers a stateless (i.e. no captures) C++ lambda
            // function as a method. Note that a lambda function must take a
            // `c10::intrusive_ptr<YourClass>` (or some const/ref version of
            // that) as the first argument. Other arguments can be whatever you
            // want.
            .def(
                "execute",
                [](const c10::intrusive_ptr<EngineClass>& self,
                   at::List<at::Tensor> inputs) {
                  return self->Execute(inputs);
                })
            // The following four lines expose methods of the
            // MyStackClass<std::string> class as-is. `torch::class_` will
            // automatically examine the argument and return types of the
            // passed-in method pointers and expose these to Python and
            // TorchScript accordingly. Finally, notice that we must take the
            // *address* of the fully-qualified method name, i.e. use the unary
            // `&` operator, due to C++ typing rules.
            .def("dump_attr_to_file", &EngineClass::DumpAttrToFile)
            .def("dump_model_proto", &EngineClass::DumpModelProto)
            .def("get_attr_string", &EngineClass::GetAttrString)
            .def("get_attr_keys", &EngineClass::GetAttrKeys)
            .def("last_inputs", &EngineClass::last_inputs)
            .def("last_outputs", &EngineClass::last_outputs)
            // class_<>::def_pickle allows you to define the serialization
            // and deserialization methods for your C++ class.
            // Currently, we only support passing stateless lambda functions
            // as arguments to def_pickle
            .def_pickle(
                // __getstate__
                // This function defines what data structure should be produced
                // when we serialize an instance of this class. The function
                // must take a single `self` argument, which is an intrusive_ptr
                // to the instance of the object. The function can return
                // any type that is supported as a return value of the
                // TorchScript custom operator API. In this instance, we've
                // chosen to return a std::vector<std::string> as the salient
                // data to preserve from the class.
                [](const c10::intrusive_ptr<EngineClass>& self)
                    -> EngineClass::SerialType { return self->Serialize(); },
                // __setstate__
                // This function defines how to create a new instance of the C++
                // class when we are deserializing. The function must take a
                // single argument of the same type as the return value of
                // `__getstate__`. The function must return an intrusive_ptr
                // to a new instance of the C++ class, initialized however
                // you would like given the serialized state.
                [](EngineClass::SerialType serialized)
                    -> c10::intrusive_ptr<EngineClass> {
                  // A convenient way to instantiate an object and get an
                  // intrusive_ptr to it is via `make_intrusive`. We use
                  // that here to allocate an instance of
                  // MyStackClass<std::string> and call the single-argument
                  // std::vector<std::string> constructor with the serialized
                  // state.
                  return EngineClass::Deserialize(std::move(serialized));
                });
  });
  return true;
}

torch::IValue create_engine(
    const EngineState& engine_state,
    const std::string& attr_debug_name,
    const std::string& fallback_module_bytes,
    const std::string& original_subgraph) {
  EngineClass::SerialType serialized = EngineClass::Serialize(
      engine_state, attr_debug_name, fallback_module_bytes, original_subgraph);

  auto custom_class_obj =
      torch::make_custom_class<EngineClass>(std::move(serialized));
  return custom_class_obj;
}

torch::TypePtr register_engine(
    torch::jit::Module& module,
    const EngineState& engine_state,
    const std::string& attr_debug_name,
    const std::string& fallback_module_bytes,
    const std::string& original_subgraph) {
  auto custom_class_obj = create_engine(
      engine_state, attr_debug_name, fallback_module_bytes, original_subgraph);
  module.register_attribute(
      attr_debug_name, custom_class_obj.type(), custom_class_obj);
  return custom_class_obj.type();
}

static bool init_dummy = InitTorchBladeEngine();
} // namespace backends
} // namespace blade
} // namespace torch
