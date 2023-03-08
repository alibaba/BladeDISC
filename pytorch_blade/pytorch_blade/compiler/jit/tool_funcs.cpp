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

/*
 ____________________________
< RAM wasn't built in a day. >
 ----------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
 */
#include "pytorch_blade/compiler/jit/tool_funcs.h"

#include <torch/script.h>
namespace torch {
namespace blade {

void unsafe_remove_method(torch::jit::Module& self, const std::string& name) {
  auto self_type = self._ivalue()->type();

  const c10::QualifiedName& method_name =
      torch::QualifiedName(*self.type()->name(), name);
  self.type()->unsafeRemoveMethod(name);
  self._ivalue()->compilation_unit()->unsafeRemoveMethod(method_name);
}

void unsafe_remove_type_attribute(
    torch::jit::Module& self,
    const std::string& name) {
  // This is used for removing the registered attribute
  // after the calibration data is collected.
  self.type()->unsafeRemoveAttribute(name);
}

torch::jit::Module clone_cpp_module(torch::jit::Module& self) {
  return torch::jit::Module(self.clone(false));
}

void create_method_from_graph(
    torch::jit::Module& self,
    const std::string& name,
    const std::shared_ptr<torch::jit::Graph>& orig_graph) {
  // The graph would be modified and passed to the compilation unit,
  // and referenced by the GraphFunction is being created
  auto graph = orig_graph->copy();
  auto self_type = self._ivalue()->type();
  bool has_input = graph->inputs().size() > 0;
  bool has_param_self = false;
  if (has_input) {
    has_param_self = graph->inputs()[0]->type() == self_type;
  }

  if (!has_param_self) {
    auto v = graph->insertInput(0, "self");
    v->setType(self_type);
  }
  const auto method_name = torch::QualifiedName(*self.type()->name(), name);
  auto fn =
      self._ivalue()->compilation_unit()->create_function(method_name, graph);
  self.type()->addMethod(fn);
}

void register_attr(
    torch::jit::Module& module,
    const std::string& name,
    torch::jit::Module& new_module) {
  module.register_attribute(name, new_module.type(), new_module._ivalue());
}

torch::jit::Node* create_get_attr_node(
    torch::jit::Graph* graph,
    torch::jit::Value* obj,
    const std::string& field) {
  auto n = graph->createGetAttr(obj, field);
  return n;
}

bool is_concrete_shape_tensor_type(const torch::jit::Value& val) {
  const auto& tensor_type = val.type()->cast<at::TensorType>();
  if (tensor_type) {
    return tensor_type->scalarType() && tensor_type->sizes().concrete_sizes();
  }

  return false;
}

bool is_gpu_tensor_type(const torch::jit::Value& val) {
  if (!val.type()->isSubtypeOf(at::TensorType::get())) {
    return false;
  }
  c10::optional<torch::Device> dev =
      val.type()->cast<at::TensorType>()->device();
  if (dev) {
    return dev->is_cuda();
  }
  return false;
}

void set_value_type(
    torch::jit::Value& val,
    const std::vector<int64_t>& shape_vec,
    const std::vector<int64_t>& stride_vec,
    const std::string& device_str,
    int64_t dtype,
    bool requires_grad,
    bool is_contiguous) {
  at::ScalarType scalar_type = static_cast<at::ScalarType>(dtype);
  torch::Device device(device_str);
  c10::VaryingShape<int64_t> sizes = shape_vec;
  c10::VaryingShape<int64_t> strides = stride_vec;
  auto type = at::TensorType::create(
      scalar_type, device, sizes, strides, requires_grad, false, is_contiguous);

  CHECK(type->requires_grad() == requires_grad);
  val.setType(type);
  CHECK(val.requires_grad() == requires_grad);
}

std::string node_schema_str(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  if (schema) {
    return c10::toString(*schema);
  } else {
    return node.kind().toQualString();
  }
}

std::string node_overload_name(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  if (schema) {
    auto overload_name = schema->overload_name();
    if (overload_name.empty()) {
      return schema->name();
    } else {
      return schema->name() + "." + overload_name;
    }
  } else {
    return node.kind().toQualString();
  }
}

torch::TypePtr fromNumberType(torch::TypePtr typ) {
  if (typ->isSubtypeOf(IntType::get())) {
    return at::TensorType::createContiguous(at::kLong, at::kCPU, {});
  } else if (typ->isSubtypeOf(FloatType::get())) {
    return at::TensorType::createContiguous(at::kFloat, at::kCPU, {});
  } else if (typ->isSubtypeOf(BoolType::get())) {
    return at::TensorType::createContiguous(at::kBool, at::kCPU, {});
  }
  return nullptr;
}

bool cast_to_i32_tensor_type(torch::jit::Value& value) {
  const auto& tensor_type = value.type()->cast<at::TensorType>();
  if (tensor_type) {
    return value.setType(tensor_type->withScalarType(at::kInt));
  } else {
    return value.setType(
        at::TensorType::createContiguous(at::kInt, at::kCPU, {}));
  }
}

bool cast_to_tensor_type(torch::jit::Value& value) {
  auto tensor_type = fromNumberType(value.type());
  if (tensor_type != nullptr) {
    value.setType(tensor_type);
    return true;
  } else {
    return false;
  }
}

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const at::Tensor& val) {
  torch::jit::Node* constant_node = g->insertNode(g->create(prim::Constant));
  constant_node->t_(attr::value, val);
  constant_node->output()->inferTypeFrom(val);
  return constant_node;
}

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const int& val) {
  torch::jit::Node* constant_node = g->insertNode(g->create(prim::Constant));
  constant_node->i_(attr::value, val);
  constant_node->output()->setType(c10::IntType::get());
  return constant_node;
}

torch::jit::Node* create_prim_constant_with_val(
    std::shared_ptr<torch::jit::Graph> g,
    const bool& val) {
  torch::jit::Node* constant_node = g->insertNode(g->create(prim::Constant));
  constant_node->i_(attr::value, val);
  constant_node->output()->setType(c10::BoolType::get());
  return constant_node;
}
} // namespace blade
} // namespace torch
