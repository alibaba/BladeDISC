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

#include "pybind_functions.h"

#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include "common_utils/utils.h"
#include "compiler/jit/torch/const_loop_unroll.h"
#include "compiler/jit/torch/eliminate_redundant_permutations.h"
#include "compiler/jit/torch/freeze_module.h"
#include "compiler/jit/torch/onnx.h"

#include "compiler/jit/fusion.h"
#include "compiler/jit/shape_type_spec.h"
#include "compiler/jit/tool_funcs.h"

namespace torch {
namespace blade {
namespace {
using torch::autograd::Variable;
using namespace torch::jit;

std::pair<std::shared_ptr<Graph>, Stack> _createGraphByTracing(
    Stack trace_inputs,
    bool strict,
    bool force_outplace,
    Module* self = nullptr) {
  C10_LOG_API_USAGE_ONCE("blade.torch.tracer");

  auto outs = ::torch::jit::tracer::trace(
      std::move(trace_inputs),
      [&](Stack inputs) -> Stack {
        auto outs = self->forward(inputs);
        return {outs};
      },
      [](const Variable& var) -> std::string { return ""; },
      strict,
      force_outplace,
      self);
  return std::make_pair(std::get<0>(outs)->graph, std::get<1>(outs));
}
} // namespace
void initToolsBindings(py::module& m) {
  py::module tools =
      m.def_submodule("_tools", "torch_blade python toolkit bindings");

  tools.def("read_bool_from_env", env::ReadBoolFromEnvVar);
  tools.def("read_double_from_env", env::ReadDoubleFromEnvVar);
  tools.def("merge_node_into_group", &torch::blade::MergeNodeIntoGroup);
  tools.def("clone_cpp_module", &torch::blade::clone_cpp_module);
  tools.def(
      "create_method_from_graph", &torch::blade::create_method_from_graph);
  tools.def("unsafe_remove_method", &torch::blade::unsafe_remove_method);
  tools.def(
      "unsafe_remove_type_attribute",
      &torch::blade::unsafe_remove_type_attribute);
  tools.def("get_list_tensor_type", &torch::blade::get_list_tensor_type);
  tools.def("set_tensor_shape", &torch::blade::set_tensor_shape);
  tools.def(
      "create_tensor_type_from_scalar_type",
      &torch::blade::create_tensor_type_from_scalar_type);
  tools.def("_jit_pass_const_loop_unrolling", &torch::blade::UnrollConstLoops);

  // PyTorch does not expose `torch::jit::Node::{isBefore,isAfter}` to
  // Python side. We add the bindings here and then patch them to
  // `torch.jit.Node` in torch_blade/__init__.py
  tools.def(
      "node_is_before",
      [](const torch::jit::Node& n1, const torch::jit::Node& n2) {
        return n1.isBefore(&n2);
      });
  tools.def(
      "node_is_after",
      [](const torch::jit::Node& n1, const torch::jit::Node& n2) {
        return n1.isAfter(&n2);
      });
  tools.def("register_attr", &torch::blade::register_attr);
  tools.def("graph_create_get_attr", &torch::blade::create_get_attr_node);

  // PyTorch torch::jit::freeze_module
  tools.def(
      "freeze_module",
      &torch::blade::freeze_module,
      py::arg("module"),
      py::arg("preservedAttrs") = std::vector<std::string>(),
      py::arg("freezeInterfaces") = true,
      py::arg("preserveParameters") = false,
      py::arg("disableShapePeephole") = true);

  tools.def("_jit_pass_onnx", torch::blade::ToONNX);
  tools.def(
      "subgraph_input_name_mangle", &torch::blade::subgraph_input_name_mangle);
  tools.def(
      "subgraph_input_name_demangle",
      &torch::blade::subgraph_input_name_demangle);
  tools.def(
      "_jit_pass_lower_simple_tuples",
      [](const std::shared_ptr<torch::jit::Graph>& graph) {
        return torch::jit::LowerSimpleTuples(graph);
      });
  tools.def("is_gpu_tensor_type", is_gpu_tensor_type);
  tools.def("is_concrete_shape_tensor_type", is_concrete_shape_tensor_type);
  tools.def("set_value_type", set_value_type);
  tools.def("node_schema_str", node_schema_str);
  tools.def("node_overload_name", node_overload_name);
  tools.def("cast_to_tensor_type", cast_to_tensor_type);
  tools.def("cast_to_i32_tensor_type", cast_to_i32_tensor_type);

  tools.def(
      "set_trust_tracing_shape",
      &SetTrustTracingShape,
      R"doc(
Setting trust tracing shape flag and return the old one.
Note this is per thread level setting, for multithreading use cases,
it needs to be configured separately in each thread.
      )doc");
  tools.def(
      "get_trust_tracing_shape",
      &GetTrustTracingShape,
      R"doc(
Getting trust tracing shape flag configured in current thread.
      )doc");
  tools.def(
      "set_record_cluster_io_flag",
      &SetRecordClusterIOFlag,
      R"doc(
Setting record cluster IO(Inputs/Outputs) flag and return the old one.
Note this is per thread level setting, for multithreading use cases,
it needs to be configured separately in each thread.
      )doc");
  tools.def(
      "get_record_cluster_io_flag",
      &GetRecordClusterIOFlag,
      R"doc(
Getting record cluster IO(Inputs/Outputs) flag configured in current thread.
      )doc");

  tools.def(
      "create_method_from_trace",
      [](Module& self,
         const std::string& name,
         const py::tuple& input_tuple,
         bool strict,
         bool force_outplace) {
        // prereq: Module's buffers and parameters are unique
        // this was ensured in python before calling this function
        auto typed_inputs = toTraceableStack(input_tuple);

        std::shared_ptr<Graph> graph = std::get<0>(
            _createGraphByTracing(typed_inputs, strict, force_outplace, &self));
        const auto method_name = QualifiedName(*self.type()->name(), name);
        auto fn = self._ivalue()->compilation_unit()->create_function(
            method_name, graph);
        self.type()->addMethod(fn);
      });
}

} // namespace blade
} // namespace torch
