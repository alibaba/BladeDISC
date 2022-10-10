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

#include "pytorch_blade/compiler/jit/fusion.h"

#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include <algorithm>
#include <unordered_map>

#include "common_utils/logging.h"
namespace torch {
namespace blade {
using namespace ::torch::jit;

const std::string subgraph_input_name_suffix = "_";

std::string subgraph_input_name_mangle(const std::string& inp_name) {
  return inp_name + subgraph_input_name_suffix;
}

std::string subgraph_input_name_demangle(const std::string& inp_name) {
  const std::string::size_type inp_len = inp_name.length();
  const std::string::size_type suffix_len = subgraph_input_name_suffix.length();
  if (inp_len < suffix_len ||
      inp_name.substr(inp_len - suffix_len, suffix_len) !=
          subgraph_input_name_suffix) {
    return inp_name;
  } else {
    return inp_name.substr(0, inp_len - suffix_len);
  }
}

// insert a producer node into a consuming fusion group.
// DOES NOT WORK if n is a consumer of an output of the fusion group
// returns the node _inside_ the group that represents the node
Node* MergeNodeIntoGroup(Node* group, Node* n) {
  AT_ASSERT(n->kind() != prim::FusionGroup);
  auto& subgraph = *group->g(attr::Subgraph);
  assert(group->inputs().size() == subgraph.inputs().size());
  // map from nodes in the surrounding graph to parameters in the fusion
  // group's subgraph that correspond to them
  std::unordered_map<Value*, Value*> inputs_map;
  size_t i = 0;
  size_t tensor_insert_idx = 0;
  for (auto input : group->inputs()) {
    inputs_map[input] = subgraph.inputs()[i++];
    if (input->type()->isSubtypeOf(TensorType::get()))
      tensor_insert_idx = i;
  }
  // add n's inputs to the fusion group's input list if we don't already have
  // them
  // we insert tensors first because the fuser assumes that to be the case
  // (as a legacy from tensors only)
  auto map_input = [](std::unordered_map<Value*, Value*>& inputs_map,
                      Value* input,
                      Value* in_group) {
    in_group->setType(input->type());
    in_group->setDebugName(subgraph_input_name_mangle(input->debugName()));
    inputs_map[input] = in_group;
  };
  WithInsertPoint guard(*subgraph.nodes().begin());
  for (auto input : n->inputs()) {
    if (inputs_map.count(input) == 0) {
      if (input->node()->kind() == prim::Constant) {
        // clone prim::Constant into subgraph if found
        Node* in_const =
            subgraph.createClone(input->node(), [](Value*) -> Value* {
              throw std::runtime_error("unexpected input in constant node");
            });
        subgraph.insertNode(in_const);
        inputs_map[input] = in_const->output();
      } else if (input->type()->isSubtypeOf(TensorType::get())) {
        auto in_group = subgraph.insertInput(tensor_insert_idx);
        map_input(inputs_map, input, in_group);
        group->insertInput(tensor_insert_idx, input);
        tensor_insert_idx++;
      } else {
        // Non-Tensor/Constant input types
        auto in_group = subgraph.addInput();
        map_input(inputs_map, input, in_group);
        group->addInput(input);
      }
    }
  }
  // copy n into the graph, remapping its inputs to internal nodes
  Node* in_graph = subgraph.createClone(n, [&](Value* k) -> Value* {
    auto ret = inputs_map[k];
    CHECK_NOTNULL(ret);
    return ret;
  });
  // if n's outputs are already inputs to the fusion group,
  // we need to remove them because n is now inside the fusion group.
  //
  // i.e.,
  // x = f(w); group(x, y, z) becomes group(w, y, z).
  // x, y, z = f(w); group(x, y, z) becomes group(w).
  //
  // remapping nodes that used the input to the newly-merged node
  // n is not an input when the fusion group is empty
  auto inputs = group->inputs();
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
    if (it != inputs.end()) {
      size_t p = it - inputs.begin();
      group->removeInput(p);
      subgraph.inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
      subgraph.eraseInput(p);
    }
  }
  return subgraph.insertNode(in_graph);
}

torch::TypePtr get_list_tensor_type() {
  auto tensor_type = at::TensorType::get();
  auto list_type = at::ListType::create(tensor_type);
  return list_type;
}

torch::TypePtr create_tensor_type_from_scalar_type(const c10::Type& typ) {
  if (typ.isSubtypeOf(c10::IntType::get())) {
    return TensorType::createContiguous(at::kLong, at::kCPU, {});
  } else if (typ.isSubtypeOf(c10::FloatType::get())) {
    return TensorType::createContiguous(at::kDouble, at::kCPU, {});
  } else if (typ.isSubtypeOf(c10::BoolType::get())) {
    return TensorType::createContiguous(at::kBool, at::kCPU, {});
  }
  return nullptr;
}

void set_tensor_shape(
    torch::jit::Value* val,
    const std::vector<int64_t>& dims) {
  auto tensor_type = val->type()->cast<c10::TensorType>();
  if (tensor_type) {
    val->setType(c10::TensorType::create(
        tensor_type->scalarType(),
        tensor_type->device(),
        c10::SymbolicShape(dims),
        c10::VaryingShape<c10::Stride>(dims.size()),
        tensor_type->requires_grad()));
    return;
  }
  TORCH_CHECK(false, "input value should be tensor type");
}

} // namespace blade
} // namespace torch
