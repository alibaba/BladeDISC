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

#include "pytorch_blade/quantization/pybind_functions.h"
#include "alias.h"
#include "pytorch_blade/compiler/jit/tool_funcs.h"

#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/script.h>
#include <iostream>

namespace torch {
namespace blade {
namespace quantization {
using namespace torch::jit;

void add_placeholder_for_fake_quant(Module& model) {
  Symbol sym = Symbol::fromQualString(
      torch::blade::quantization::custom_placeholder_name);
  auto g = model.get_method("forward").graph();
  // the graph should be inlined first
  Inline(*g);
  // add a placeholder op after each aten::fake_quantize_per_channel_affine
  // node, which is to prevent the aten::fake_quantize_per_channel_affine be
  // folded by the ConstantPropagation pass.
  for (auto n : g->nodes()) {
    if (n->kind().toQualString() ==
        std::string(torch::blade::quantization::
                        at_fake_quant_per_channel_affine_name)) {
      auto place_holder = g->appendNode(g->create(sym));
      place_holder->moveAfter(n);
      n->outputs()[0]->replaceAllUsesWith(place_holder->outputs()[0]);
      place_holder->addInput(n->outputs()[0]);
    }
  }
}

void remove_placeholder(Module& model) {
  auto g = model.get_method("forward").graph();
  std::vector<Node*> place_holder_nodes;
  for (auto n : g->nodes()) {
    if (n->kind().toQualString() ==
        std::string(torch::blade::quantization::custom_placeholder_name)) {
      n->outputs()[0]->replaceAllUsesWith(n->inputs()[0]);
      n->removeAllInputs();
      place_holder_nodes.push_back(n);
    }
  }
  for (auto n : place_holder_nodes) {
    n->destroy();
  }
}

void replace_aten_fake_quant_with_custom_version(Module& model) {
  auto g = model.get_method("forward").graph();
  // the graph should be inlined first
  Inline(*g);
  Symbol sym = Symbol::fromQualString(
      torch::blade::quantization::custom_fake_quant_name);
  std::vector<Node*> aten_fake_quant_node;

  for (auto n : g->nodes()) {
    std::string node_kind_str = n->kind().toQualString();
    if (node_kind_str ==
            std::string(torch::blade::quantization::
                            at_fake_quant_per_channel_affine_name) ||
        node_kind_str ==
            std::string(torch::blade::quantization::
                            at_fake_quant_per_tensor_affine_name)) {
      // aten::fake_quant will be destroyed later, collect it here
      aten_fake_quant_node.push_back(n);

      // create torch_blade_fake_quant custom op
      Node* fake_quant_node = g->insertNode(g->create(sym));
      fake_quant_node->moveAfter(n);
      fake_quant_node->output()->setType(n->inputs()[0]->type());
      n->output()->replaceAllUsesWith(fake_quant_node->output());

      Value* inputs = n->inputs()[0]; // inputs

      // scale & zero_zero point could be scalar in
      // aten::fake_quantize_per_tensor_affine. Convert it to tensor to match
      // our schema.
      Value* scales = n->inputs()[1];
      if (!scales->type()->isSubtypeOf(at::TensorType::get())) {
        // create a
        TORCH_CHECK(
            scales->type()->isSubtypeOf(c10::NumberType::get()),
            "Unexpected scalar type for scales.");
        float scale_num = scales->node()->f(attr::value);
        Tensor scale_tensor = torch::tensor(scale_num, torch::kFloat);
        scales = insert_prim_constant<torch::Tensor>(
            g, scales->node(), true, scale_tensor);
      }

      Value* zero_points = n->inputs()[2];
      if (!zero_points->type()->isSubtypeOf(at::TensorType::get())) {
        TORCH_CHECK(
            zero_points->type()->isSubtypeOf(c10::NumberType::get()),
            "Unexpected scalar type for zero point");
        int zp_num = zero_points->node()->i(attr::value);
        Tensor zp_tensor = torch::tensor(zp_num, torch::kInt);
        zero_points = insert_prim_constant<torch::Tensor>(
            g, zero_points->node(), true, zp_tensor);
      }

      Value* quant_min;
      Value* quant_max;
      Node* list_axis_node;
      Value* use_per_channel;
      if (node_kind_str ==
          std::string(torch::blade::quantization::
                          at_fake_quant_per_channel_affine_name)) {
        Value* axis = n->inputs()[3];
        // Just construct a prim::ListConstruct which return an empty
        // vector when the quantization scheme is per-tensor.
        list_axis_node = g->insertNode(g->create(prim::ListConstruct));
        list_axis_node->addInput(axis);
        list_axis_node->moveAfter(axis->node());
        list_axis_node->output()->setType(c10::ListType::ofInts());
        quant_min = n->inputs()[4];
        quant_max = n->inputs()[5];
        use_per_channel = insert_prim_constant<bool>(g, n, false, true);
      } else {
        // According to:
        // https://github.com/pytorch/pytorch/blob/7134b9bc7b1d25b453ec5c53b1ec70cb206228a1/torch/csrc/jit/ir/constants.cpp#L211
        // Can not directly construct a prim::Constant using constant literal of
        // type List[int]. So use prim::ListConstruct the obtain the axis in the
        // required format.
        list_axis_node = g->insertNode(g->create(prim::ListConstruct));
        list_axis_node->moveBefore(n);
        list_axis_node->output()->setType(c10::ListType::ofInts());
        quant_min = n->inputs()[3];
        quant_max = n->inputs()[4];
        use_per_channel = insert_prim_constant<bool>(g, n, false, false);
      }

      // For now, only 8 bit quantization is supported
      Value* num_bits = insert_prim_constant<int>(g, n, false, 8);

      Value* use_signed;
      Value* use_symmetric;
      int quant_min_num = quant_min->node()->i(attr::value);
      int quant_max_num = quant_max->node()->i(attr::value);
      if ((quant_min_num ^ quant_max_num) < 0) {
        use_signed = insert_prim_constant<bool>(g, n, false, true);
        use_symmetric = insert_prim_constant<bool>(g, n, false, true);
      } else {
        use_signed = insert_prim_constant<bool>(g, n, false, false);
        use_symmetric = insert_prim_constant<bool>(g, n, false, false);
      }

      // aten::fake_quant is aimed for static quantization.
      // So use_dynamic is set to false.
      Value* use_dynamic = insert_prim_constant<bool>(g, n, false, false);
      // add all prepared jit::Values to the new constructed fake_quant node
      fake_quant_node->addInput(inputs);
      fake_quant_node->addInput(scales);
      fake_quant_node->addInput(zero_points);
      fake_quant_node->addInput(quant_min);
      fake_quant_node->addInput(quant_max);
      fake_quant_node->addInput(num_bits);
      fake_quant_node->addInput(list_axis_node->output());
      fake_quant_node->addInput(use_signed);
      fake_quant_node->addInput(use_symmetric);
      fake_quant_node->addInput(use_dynamic);
      fake_quant_node->addInput(use_per_channel);
    }
  }
  for (auto n : aten_fake_quant_node) {
    n->destroy();
  }

  // some jit passes to clean the graph
  EliminateDeadCode(g->block());
}

void initQuantizationBindings(py::module& m) {
  py::module quantization = m.def_submodule(
      "_quantization", "torch_blade python bindings for quantization");
  quantization.def(
      "add_placeholder_for_fake_quant", &add_placeholder_for_fake_quant);
  quantization.def("remove_placeholder", &remove_placeholder);
  quantization.def(
      "replace_aten_fake_quant_with_custom_version",
      &replace_aten_fake_quant_with_custom_version);
  quantization.attr("at_fake_quant_per_tensor_affine_name") =
      torch::blade::quantization::at_fake_quant_per_tensor_affine_name;
  quantization.attr("at_fake_quant_per_channel_affine_name") =
      torch::blade::quantization::at_fake_quant_per_channel_affine_name;
}

} // namespace quantization
} // namespace blade
} // namespace torch
