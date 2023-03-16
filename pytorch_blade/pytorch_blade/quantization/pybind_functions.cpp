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
#include "pytorch_blade/compiler/jit/tool_funcs.h"
#include "pytorch_blade/quantization/alias.h"

#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/script.h>
#include <cmath>
#include <unordered_set>

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
                            at_fake_quant_per_channel_affine_name) ||
        n->kind().toQualString() ==
            std::string(torch::blade::quantization::custom_fake_quant_name)) {
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

template <typename T>
torch::jit::Value* insert_prim_constant(
    std::shared_ptr<torch::jit::Graph> g,
    torch::jit::Node* n,
    bool is_after,
    const T& val) {
  torch::jit::Node* constant_node = create_prim_constant_with_val(g, val);
  if (is_after) {
    constant_node->moveAfter(n);
  } else {
    constant_node->moveBefore(n);
  }
  return constant_node->output();
}

const static std::unordered_set<std::string> weight_only_list{"aten::linear"};

// Prepare graph for weight-only quantization.
// 1. extract the weight of each weight-only quantizable op
// 2. calculate the scale & zero_point of each weight. (currently use min/max
// observer)
// 3. add a torch_blade.fake_quant to each weight
// NOTE: This pass assumes that the graph is frozen.
// In other words, the weight tensor of aten op is from
// prim::Constant op instead of prim::GetAttr.
// TODO: Currently, there is not difference between the fake-quant used in
// static quantization and that used in weight-only quantization. This will
// make it impossible for use to distinguish between the two when doing mix
// type quantization. (e.g. use static and weight-only quantization
// simultaneously) Consider to add a new attribute like weight-only for
// torch_blade::fake_quant
void add_fake_quant_for_weight(Module& model) {
  auto g = model.get_method("forward").graph();
  Symbol sym = Symbol::fromQualString(
      torch::blade::quantization::custom_fake_quant_name);

  for (auto&& n : g->nodes()) {
    std::string node_kind_str = n->kind().toQualString();
    if (weight_only_list.find(node_kind_str) != weight_only_list.end()) {
      // TODO: If support quantizing other types of layers, check whether index
      // 1 meets the requirements.
      Value* weight_val = n->inputs()[1];
      auto weight_val_type = weight_val->type()->cast<c10::TensorType>();
      if (!weight_val_type) {
        // The probability of this condition being triggered is very small,
        // however, for safety reasons, we still do this check.
        continue;
      }

      Node* weight_node = weight_val->node();
      if (weight_node->kind() != prim::Constant) {
        // So the graph should be frozen
        continue;
      }
      c10::optional<IValue> constant = weight_node->t(attr::value);
      const at::Tensor& weight_t = constant->toIValue().toTensor();
      at::Tensor weight_min_t, weight_max_t;

      // TODO: determine dim according to the type of layer to be quantized
      std::tie(weight_min_t, weight_max_t) = at::_aminmax(weight_t, 1);

      // TODO: calculate the quantization info based on the backend
      int32_t quant_min = -128;
      int32_t quant_max = 127;
      // the following process is same on it in
      // UniformQuantizationObserverBase's _calculate_qparams
      at::Tensor min_val_neg_t =
          torch::min(weight_min_t, torch::zeros_like(weight_min_t));
      at::Tensor max_val_pos_t =
          torch::max(weight_max_t, torch::zeros_like(weight_max_t));
      auto device = weight_val_type->device();
      auto scale_option =
          torch::TensorOptions().dtype(torch::kFloat32).device(device);
      at::Tensor scale_t = torch::ones(min_val_neg_t.sizes(), scale_option);
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 10
      auto zero_point_option =
          torch::TensorOptions().dtype(torch::kInt32).device(device);
#else
      auto zero_point_option =
          torch::TensorOptions().dtype(torch::kInt64).device(device);
#endif
      at::Tensor zero_point_t =
          torch::zeros(min_val_neg_t.sizes(), zero_point_option);
      // for per_channel_symmetric
      max_val_pos_t = torch::max(-min_val_neg_t, max_val_pos_t);
      scale_t = max_val_pos_t / (float(quant_max - quant_min) / 2);
      const static float epsilon = std::numeric_limits<float>::epsilon();
      at::Tensor epsilon_t = torch::ones_like(scale_t) * epsilon;
      scale_t = torch::max(scale_t, epsilon_t);

      // Create torch_blade.fake_quant for the weight,
      // and replace the origin weight input with the output
      // of the new constructed fake_quant node
      Node* fake_quant_node = g->insertNode(g->create(sym));
      fake_quant_node->moveAfter(weight_node);
      fake_quant_node->output()->setType(weight_node->outputs()[0]->type());
      n->replaceInputWith(weight_val, fake_quant_node->outputs()[0]);

      // Create needed inputs to the torch_blade.fake_quant
      // 1. scale
      Value* scale_val =
          insert_prim_constant(g, fake_quant_node, false, scale_t);

      // 2. zero_point
      Value* zero_point_val =
          insert_prim_constant(g, fake_quant_node, false, zero_point_t);

      // 3. quant_min & quant_max
      Value* quant_min_val =
          insert_prim_constant<int>(g, fake_quant_node, false, quant_min);
      Value* quant_max_val =
          insert_prim_constant<int>(g, fake_quant_node, false, quant_max);

      // 4. num_bits
      // TODO: support more kinds of num_bits
      Value* num_bits_val =
          insert_prim_constant<int>(g, fake_quant_node, false, 8);

      // 5. axis
      Value* axis_val = insert_prim_constant<int>(g, fake_quant_node, false, 0);
      Node* list_axis_node = g->insertNode(g->create(prim::ListConstruct));
      list_axis_node->addInput(axis_val);
      list_axis_node->moveAfter(axis_val->node());
      list_axis_node->output()->setType(c10::ListType::ofInts());

      // 6. boolean value
      Value* use_signed_val =
          insert_prim_constant<bool>(g, fake_quant_node, false, true);
      Value* use_symmetric_val =
          insert_prim_constant<bool>(g, fake_quant_node, false, true);
      Value* use_dynamic_val =
          insert_prim_constant<bool>(g, fake_quant_node, false, true);
      Value* use_per_channel_val =
          insert_prim_constant<bool>(g, fake_quant_node, false, true);

      fake_quant_node->addInput(weight_val);
      fake_quant_node->addInput(scale_val);
      fake_quant_node->addInput(zero_point_val);
      fake_quant_node->addInput(quant_min_val);
      fake_quant_node->addInput(quant_max_val);
      fake_quant_node->addInput(num_bits_val);
      fake_quant_node->addInput(list_axis_node->output());
      fake_quant_node->addInput(use_signed_val);
      fake_quant_node->addInput(use_symmetric_val);
      fake_quant_node->addInput(use_dynamic_val);
      fake_quant_node->addInput(use_per_channel_val);
    }
  }
  // some jit passes to clean the graph
  EliminateDeadCode(g->block());
}

void replace_aten_fake_quant_with_custom_version(Module& model) {
  auto g = model.get_method("forward").graph();
  // the graph should be inlined first
  Inline(*g);
  Symbol sym = Symbol::fromQualString(
      torch::blade::quantization::custom_fake_quant_name);
  std::vector<Node*> aten_fake_quant_node;

  for (auto&& n : g->nodes()) {
    std::string node_kind_str = n->kind().toQualString();

    if (node_kind_str ==
            std::string(torch::blade::quantization::
                            at_fake_quant_per_channel_affine_name) ||
        node_kind_str ==
            std::string(torch::blade::quantization::
                            at_fake_quant_per_tensor_affine_name)) {
      // aten::fake_quant will be destroyed later, collect it here
      aten_fake_quant_node.push_back(n);

      auto input_type = n->inputs()[0]->type()->cast<c10::TensorType>();
      TORCH_CHECK(input_type, "Unexpected input type of aten::fake_quant.")
      auto device = input_type->device();

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
        TORCH_CHECK(
            scales->type()->isSubtypeOf(c10::NumberType::get()),
            "Unexpected scalar type for scales.");
        float scale_num = scales->node()->f(attr::value);
        auto tensor_option =
            torch::TensorOptions().dtype(torch::kFloat).device(device);
        Tensor scale_tensor = torch::tensor(scale_num, tensor_option);
        scales = insert_prim_constant(g, scales->node(), true, scale_tensor);
      }

      Value* zero_points = n->inputs()[2];
      if (!zero_points->type()->isSubtypeOf(at::TensorType::get())) {
        TORCH_CHECK(
            zero_points->type()->isSubtypeOf(c10::NumberType::get()),
            "Unexpected scalar type for zero point");
        int zp_num = zero_points->node()->i(attr::value);
        auto tensor_option =
            torch::TensorOptions().dtype(torch::kInt).device(device);
        Tensor zp_tensor = torch::tensor(zp_num, tensor_option);
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

      Value* use_signed;
      Value* use_symmetric;
      int64_t quant_min_num = quant_min->node()->i(attr::value);
      int64_t quant_max_num = quant_max->node()->i(attr::value);
      if ((quant_min_num ^ quant_max_num) < 0) {
        use_signed = insert_prim_constant<bool>(g, n, false, true);
        use_symmetric = insert_prim_constant<bool>(g, n, false, true);
      } else {
        use_signed = insert_prim_constant<bool>(g, n, false, false);
        use_symmetric = insert_prim_constant<bool>(g, n, false, false);
      }
      int num_bits_val = int(std::round(log2(quant_max_num - quant_min_num)));
      Value* num_bits = insert_prim_constant<int>(g, n, false, num_bits_val);

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
  quantization.def("add_fake_quant_for_weight", &add_fake_quant_for_weight);
  quantization.def("remove_placeholder", &remove_placeholder);
  quantization.def(
      "replace_aten_fake_quant_with_custom_version",
      &replace_aten_fake_quant_with_custom_version);
  quantization.attr("at_fake_quant_per_tensor_affine_name") =
      torch::blade::quantization::at_fake_quant_per_tensor_affine_name;
  quantization.attr("at_fake_quant_per_channel_affine_name") =
      torch::blade::quantization::at_fake_quant_per_channel_affine_name;
  quantization.attr("torch_blade_fake_quant_name") =
      torch::blade::quantization::custom_fake_quant_name;
}

} // namespace quantization
} // namespace blade
} // namespace torch
