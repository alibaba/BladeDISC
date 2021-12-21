/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Adopted from tensorflow/compiler/tf2xla/tf2xla_util.cc

#include "tao_bridge/tf/tf2xla_util.h"

#include <queue>
#include <random>
#include <set>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

// TODO(b/77601805): add tests for associated function related stuff.
bool HasAssociatedFunction(const NodeDef& node_def,
                           FunctionLibraryRuntime* flr) {
  if (flr->GetFunctionLibraryDefinition()->Contains(node_def.op())) {
    return true;
  }

  if (node_def.op() == FunctionLibraryDefinition::kGradientOp) {
    // Gradient op has "f" attr, which is set to the function we are getting
    // gradient for. We need to functionalize the gradient function.
    return true;
  }

  for (const auto& iter : node_def.attr()) {
    if (iter.second.has_func()) {
      return true;
    }
  }

  return false;
}

std::vector<AssociatedFunctionInfo> GetAssociatedFunctions(
    const Node& node, FunctionLibraryRuntime* flr) {
  std::vector<AssociatedFunctionInfo> results;
  const string& op = node.type_string();
  if (flr->GetFunctionLibraryDefinition()->Contains(op)) {
    // This is a function call node.
    AttrValueMap attrs(node.attrs().begin(), node.attrs().end());
    results.emplace_back(AssociatedFunctionInfo::FunctionCall(op, attrs));
  } else if (node.type_string() == FunctionLibraryDefinition::kGradientOp) {
    // This is a SymbolicGradient op.
    AttrValueMap attrs(node.attrs().begin(), node.attrs().end());
    results.emplace_back(AssociatedFunctionInfo::SymbolicGradient(op, attrs));
  } else {
    // Collect all function attrs for the node.
    for (auto& iter : node.attrs()) {
      if (iter.second.has_func()) {
        VLOG(2) << "Found function attr for node " << node.name() << ": "
                << iter.first << " = " << iter.second.func().name();
        results.emplace_back(AssociatedFunctionInfo::FunctionAttr(
            iter.second.func().name(), iter.second.func().attr(), iter.first));
      }
    }
  }
  return results;
}

Status RewriteAssociatedFunction(
    Graph* graph, Node* node, FunctionLibraryDefinition* fld,
    const AssociatedFunctionInfo& associated_function,
    const string& rewritten_function_name) {
  switch (associated_function.type()) {
    case AssociatedFunctionInfo::kFunctionCallNode: {
      // Change this node to call the new function.
      NodeDefBuilder builder(node->name(), rewritten_function_name, fld);
      for (auto attr : node->attrs()) {
        builder.Attr(attr.first, attr.second);
      }
      for (int i = 0; i < node->num_inputs(); i++) {
        Node* input_node;
        TF_RETURN_IF_ERROR(node->input_node(i, &input_node));
        builder.Input(input_node->name(), i, node->input_type(i));
      }
      builder.Device(node->assigned_device_name().empty()
                         ? node->requested_device()
                         : node->assigned_device_name());
      NodeDef node_def;
      TF_RETURN_IF_ERROR(builder.Finalize(&node_def));
      Status s;
      Node* new_node = graph->AddNode(node_def, &s);
      TF_RETURN_IF_ERROR(s);
      for (auto edge : node->in_edges()) {
        graph->AddEdge(edge->src(), edge->src_output(), new_node,
                       edge->dst_input());
      }
      for (auto edge : node->out_edges()) {
        graph->AddEdge(new_node, edge->src_output(), edge->dst(),
                       edge->dst_input());
      }
      graph->RemoveNode(node);
      break;
    }
    case AssociatedFunctionInfo::kSymbolicGradient: {
      NameAttrList func;
      TF_RETURN_IF_ERROR(GetNodeAttr(
          node->attrs(), FunctionLibraryDefinition::kFuncAttr, &func));
      GradientDef gradient_def;
      gradient_def.set_function_name(func.name());
      gradient_def.set_gradient_func(rewritten_function_name);
      string original_grad_func = fld->FindGradient(func.name());
      if (original_grad_func.empty()) {
        TF_RETURN_IF_ERROR(fld->AddGradientDef(gradient_def));
      } else if (original_grad_func != rewritten_function_name) {
        TF_RETURN_IF_ERROR(fld->ReplaceGradient(gradient_def));
      }
      break;
    }
    case AssociatedFunctionInfo::kFunctionAttr: {
      // Change function attr to rewritten functions.
      NameAttrList func;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(node->attrs(), associated_function.attr_name(), &func));
      node->ClearAttr(associated_function.attr_name());
      func.set_name(rewritten_function_name);
      node->AddAttr(associated_function.attr_name(), func);
      break;
    }
  }

  return Status::OK();
}

}  // namespace tao
}  // namespace tensorflow
