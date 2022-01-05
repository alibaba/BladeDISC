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

#include "tao_bridge/tao_util.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tao {
namespace util {

bool HasOpType(const Graph &graph, absl::string_view op_type) {
  for (auto n : graph.op_nodes()) {
    if (n->type_string() == op_type) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<FunctionLibraryDefinition>
ReachableDefinitions(const FunctionLibraryDefinition &flib,
                     const FunctionDef &entry_func) {
  // Functions that are reachable from the optimized graph.
  std::unordered_set<string> keep_funcs;

  // Insert the entry func itself
  keep_funcs.insert(entry_func.signature().name());

  std::vector<const FunctionDef *> worklist;
  worklist.reserve(flib.num_functions());

  // Add registered and not already processed functions to the queue by name.
  const auto add_to_worklist = [&](const string &func_name) {
    const FunctionDef *func = flib.Find(func_name);
    if (func && keep_funcs.find(func_name) == keep_funcs.end()) {
      worklist.push_back(func);
    }
  };

  // Find all the functions that are reachable from the given node.
  const auto add_node_to_worklist = [&](const NodeDef &node) {
    // Node itself can be a call to the function.
    add_to_worklist(node.op());

    // Or node can have an attribute referencing a function.
    for (const auto &attr : node.attr()) {
      const auto &attr_value = attr.second;

      // 1. AttrValue.func
      if (attr_value.has_func()) {
        add_to_worklist(attr_value.func().name());
      }

      // 2. AttrValue.ListValue.func
      if (attr_value.has_list()) {
        for (const auto &func : attr_value.list().func()) {
          add_to_worklist(func.name());
        }
      }
    }
  };

  const auto &graph_nodes = entry_func.node_def();
  std::for_each(graph_nodes.begin(), graph_nodes.end(), add_node_to_worklist);

  // Process all reachable functions.
  while (!worklist.empty()) {
    const FunctionDef *func = worklist.back();
    worklist.pop_back();

    const string &func_name = func->signature().name();
    keep_funcs.insert(func_name);

    // Find all the functions called from the function body.
    const auto &func_body = func->node_def();
    std::for_each(func_body.begin(), func_body.end(), add_node_to_worklist);

    // Check if the function has a registered gradient.
    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty())
      add_to_worklist(grad_func_name);
  }

  FunctionDefLibrary lib;
  for (const string &func_name : keep_funcs) {
    const FunctionDef *func = CHECK_NOTNULL(flib.Find(func_name));
    *lib.add_function() = *func;

    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) {
      GradientDef *gd = lib.add_gradient();
      gd->set_function_name(func_name);
      gd->set_gradient_func(grad_func_name);
    }
  }

  std::unique_ptr<FunctionLibraryDefinition> reachable_flib;
  reachable_flib.reset(
      new FunctionLibraryDefinition(flib.default_registry(), std::move(lib)));
  return reachable_flib;
}

} // namespace util
} // namespace tao
} // namespace tensorflow
