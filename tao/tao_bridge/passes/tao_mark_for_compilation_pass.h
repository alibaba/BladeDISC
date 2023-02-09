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

// An optimization passes that marks nodes that are to be compiled with
// attribute kXlaClusterAttr. Nodes with the same cluster ID will be compiled
// together.

#ifndef TAO_TAO_BRIDGE_PASSES_TAO_MARK_FOR_COMPILATION_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_MARK_FOR_COMPILATION_PASS_H_
#include <unordered_map>
#include <unordered_set>

#include "absl/types/optional.h"
#include "tao_bridge/passes/tao_optimization_pass.h"
#include "tao_bridge/tf/compilability_check_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace tao {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks nodes in a cluster to be placed outside the xla
// compilation by the encapsulate subgraphs pass.
extern const char* const kXlaOutsideCompilationAttr;

// Marks a subset of nodes in the graph which are to be clustered
// with an attribute _XlaCluster=<cluster id> so they are picked up by the
// EncapsulateSubgraphsPass.
class TaoMarkForCompilationPass : public GraphOptimizationPass {
 public:
  TaoMarkForCompilationPass() = default;

  void set_opts(const std::unique_ptr<TaoPassOptions>& opt) {
    if (opt) {
      override_tf_xla_ops_to_cluster_ = opt->override_tf_xla_ops_to_cluster;
      if (opt->min_cluster_size > 0) {
        min_cluster_size_ = opt->min_cluster_size;
      }
      if (opt->max_cluster_size > 0) {
        max_cluster_size_ = opt->max_cluster_size;
      }
      is_inner_ = opt->inner_tao_launch;
      if (opt->feats) {
        graph_tag_ = opt->feats->Tag();
      }
    }
  }

  Status Run(const GraphOptimizationPassOptions& options) override;
  Status ResetClusterNumber();

 private:
  Status RunForTest(const GraphOptimizationPassOptions& options,
                    bool disable_deadness_analysis);

  friend class MarkForCompilationPassTestHelper;

  absl::optional<std::string> override_tf_xla_ops_to_cluster_;
  absl::optional<int> min_cluster_size_;
  absl::optional<int> max_cluster_size_;
  absl::optional<bool> is_inner_;
  std::string graph_tag_{"NA"};
};

// Returns true iff 'ndef' is a call to a function that is compilable.  A
// function is compilable iff every operator in the function body is
// compilable. If 'ndef' is not compilable and 'uncompilable_node_info' is not
// null, we will populate 'uncompilable_node_info' with uncompilable node info.
bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef,
                  RecursiveCompilabilityChecker::UncompilableNodesMap*
                      uncompilable_node_info = nullptr);

std::unordered_map<string, std::vector<string>>* GetWhitelistTable();
std::unordered_set<string>* GetFusedOpTable();

namespace testing {
// DO NOT USE IN PRODUCTION.
//
// Resets some internal state to let us write reliable unit tests.
void ResetClusterSequenceNumber();

// Return a list of operation that we choose not to put into the whitelist.
std::unordered_set<string> GetKnownXLAWhitelistOp();
}  // namespace testing
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_TAO_MARK_FOR_COMPILATION_PASS_H_
