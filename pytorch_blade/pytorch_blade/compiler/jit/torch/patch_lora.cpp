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

#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include <algorithm>
#include <unordered_map>

namespace torch {
namespace blade {
using namespace ::torch::jit;

Value* findBeforeHeadToBatchDimNode(Value* v) {
  return v->node()->input(0)->node()->input(0)->node()->input(0);
}
Value* maybeAtenToOrUnpack(Value* v) {
  if (v->node()->kind() == aten::to) {
    return v;
  } else if (v->node()->kind() == prim::TupleUnpack) {
    return v->node()->input(0)->node()->input(0);
  } else {
    std::cout << "maybeAtenToOrUnpack: not aten::to or prim::TupleUnpack: "
              << v->node()->kind().toQualString() << std::endl;
    return nullptr;
  }
}
bool markLoraInputs(Block* b) {
  int64_t weight_index = 0;
  for (Node* dropoutNode : b->nodes()) {
    if (dropoutNode->kind() == aten::dropout &&
        dropoutNode->input(0)->node()->kind() == aten::add &&
        dropoutNode->input(0)->node()->input(1)->node()->kind() == aten::mul &&
        dropoutNode->input(0)->node()->input(0)->node()->kind() ==
            aten::linear) {
      std::cout << "find lora node" << std::endl;

      auto bmmNode =
          findBeforeHeadToBatchDimNode(
              dropoutNode->input(0)->node()->input(0)->node()->input(0))
              ->node();
      auto baddbmmNode =
          bmmNode->input(0)->node()->input(0)->node()->input(0)->node();

      auto to_out_up = dropoutNode->input(0)
                           ->node()
                           ->input(1)
                           ->node()
                           ->input(0)
                           ->node()
                           ->input(0)
                           ->node();
      auto to_out_down = to_out_up->input(0)->node();

      auto to_v_up =
          maybeAtenToOrUnpack(findBeforeHeadToBatchDimNode(bmmNode->input(1))
                                  ->node()
                                  ->input(1)
                                  ->node()
                                  ->input(0))
              ->node()
              ->input(0)
              ->node();
      auto to_v_down = to_v_up->input(0)->node();

      auto to_q_up = maybeAtenToOrUnpack(
                         findBeforeHeadToBatchDimNode(baddbmmNode->input(1))
                             ->node()
                             ->input(1)
                             ->node()
                             ->input(0))
                         ->node()
                         ->input(0)
                         ->node();
      auto to_q_down = to_q_up->input(0)->node();
      auto to_k_up =
          findBeforeHeadToBatchDimNode(baddbmmNode->input(2)->node()->input(0))
              ->node()
              ->input(1)
              ->node()
              ->input(0)
              ->node()
              ->input(0)
              ->node()
              ->input(0)
              ->node()
              ->input(0)
              ->node();
      auto to_k_down = to_k_up->input(0)->node();

      std::cout << "to_out_up: " << to_out_up->kind().toQualString()
                << std::endl;
      std::cout << "to_out_down: " << to_out_down->kind().toQualString()
                << std::endl;
      std::cout << "to_v_up: " << to_v_up->kind().toQualString() << std::endl;
      std::cout << "to_v_down: " << to_v_down->kind().toQualString()
                << std::endl;
      std::cout << "to_q_up: " << to_q_up->kind().toQualString() << std::endl;
      std::cout << "to_q_down: " << to_q_down->kind().toQualString()
                << std::endl;
      std::cout << "to_k_up: " << to_k_up->kind().toQualString() << std::endl;
      std::cout << "to_k_down: " << to_k_down->kind().toQualString()
                << std::endl;
      std::cout << "bmmNode: " << bmmNode->kind().toQualString() << std::endl;
      std::cout << "baddbmmNode: " << baddbmmNode->kind().toQualString()
                << std::endl;

      std::string index = std::to_string(weight_index);

      auto to_q_down_weights = b->addInput("to_q_down_weights_" + index)
                                   ->setType(to_q_down->input(1)->type());
      auto to_q_up_weights = b->addInput("to_q_up_weights_" + index)
                                 ->setType(to_q_up->input(1)->type());

      to_q_down->input(1)->replaceAllUsesWith(to_q_down_weights);
      to_q_up->input(1)->replaceAllUsesWith(to_q_up_weights);

      auto to_k_down_weights = b->addInput("to_k_down_weights_" + index)
                                   ->setType(to_k_down->input(1)->type());
      auto to_k_up_weights = b->addInput("to_k_up_weights_" + index)
                                 ->setType(to_k_up->input(1)->type());
      to_k_down->input(1)->replaceAllUsesWith(to_k_down_weights);
      to_k_up->input(1)->replaceAllUsesWith(to_k_up_weights);

      auto to_v_down_weights = b->addInput("to_v_down_weights_" + index)
                                   ->setType(to_v_down->input(1)->type());
      auto to_v_up_weights = b->addInput("to_v_up_weights_" + index)
                                 ->setType(to_v_up->input(1)->type());
      to_v_down->input(1)->replaceAllUsesWith(to_v_down_weights);
      to_v_up->input(1)->replaceAllUsesWith(to_v_up_weights);

      auto to_out_down_weights = b->addInput("to_out_down_weights_" + index)
                                     ->setType(to_out_down->input(1)->type());
      auto to_out_up_weights = b->addInput("to_out_up_weights_" + index)
                                   ->setType(to_out_up->input(1)->type());
      to_out_down->input(1)->replaceAllUsesWith(to_out_down_weights);
      to_out_up->input(1)->replaceAllUsesWith(to_out_up_weights);

      weight_index++;
    }
  }
  return true;
}

bool markLoraInputs(Graph* graph) {
  markLoraInputs(graph->block());
  graph->lint();
  return true;
}
} // namespace blade
} // namespace torch
