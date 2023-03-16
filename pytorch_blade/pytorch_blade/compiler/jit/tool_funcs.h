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

#include <memory>
#include <string>
#include <vector>

namespace at {
class Tensor;
} // namespace at

namespace torch {
namespace jit {
class Module;
class Graph;
class Node;
class Value;
} // namespace jit

namespace blade {
void RegisterShapeRecordOp();
void unsafe_remove_method(torch::jit::Module& self, const std::string& name);
void unsafe_remove_type_attribute(
    torch::jit::Module& self,
    const std::string& name);

torch::jit::Module clone_cpp_module(torch::jit::Module& self);

// A tool to create method from a TorchScript graph,
// and add it to ScriptModule.
void create_method_from_graph(
    torch::jit::Module& self,
    const std::string& name,
    const std::shared_ptr<torch::jit::Graph>& graph);

void register_attr(
    torch::jit::Module& module,
    const std::string& name,
    torch::jit::Module& new_module);

torch::jit::Node* create_get_attr_node(
    torch::jit::Graph* graph,
    torch::jit::Value* obj,
    const std::string& field);

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const at::Tensor& val);

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const int& val);

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const bool& val);

bool is_concrete_shape_tensor_type(const torch::jit::Value& val);
bool is_gpu_tensor_type(const torch::jit::Value& val);

void set_value_type(
    torch::jit::Value& val,
    const std::vector<int64_t>& shape_vec,
    const std::vector<int64_t>& stride_vec,
    const std::string& device_str,
    int64_t scalar_type,
    bool require_grad,
    bool is_contiguous);

std::string node_schema_str(const torch::jit::Node& node);
std::string node_overload_name(const torch::jit::Node& node);
bool cast_to_i32_tensor_type(torch::jit::Value& value);
bool cast_to_tensor_type(torch::jit::Value& value);
} // namespace blade
} // namespace torch
