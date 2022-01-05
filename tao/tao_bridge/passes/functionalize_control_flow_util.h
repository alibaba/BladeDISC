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

// Adopted from tensorflow/compiler/tf2xla/functionalize_control_flow_util.h

#ifndef TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_
#define TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_

#include "absl/strings/str_join.h"
#include "tao_bridge/tf/statusor.h"
#include "tensorflow/core/graph/graph.h"

// Utility functions shared between functionalize cond and while.

namespace tensorflow {
namespace tao {

// Check that the graph has no cycle containing the given node.
Status CheckNodeNotInCycle(const Node* node, const int num_nodes);

// Comparison function used for sorting nodes consistently.
// a) resource variables are last, and
// b) sort lexicographically by name (for deterministic output).
struct NodeCmpByNameResourcesLast {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

// Returns the Node* created from the NodeDef in the Graph.
xla::tao::StatusOr<Node*> AddNodeDefToGraph(const NodeDef& node_def,
                                            Graph* graph);

// Build a retval node of given type and index.
xla::tao::StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type,
                                          int index);

// Returns a textual representation of the names of the nodes in the input.
template <typename T>
string NodesToString(const T& nodes) {
  return absl::StrCat("{",
                      absl::StrJoin(nodes, ",",
                                    [](string* output, const Node* node) {
                                      absl::StrAppend(output, node->name());
                                    }),
                      "}");
}

// Check if there are any assigned device placement.
// Functionalize only when:
// 1, no assigned placement on CPU.
// 2, all of the assigned placement are on the same GPU
bool ShouldFunctionalizeForGPU(const Graph& graph, string& device_name);

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_
