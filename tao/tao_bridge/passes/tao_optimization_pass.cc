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

#include "tao_bridge/passes/tao_optimization_pass.h"

#include <algorithm>

#include "tao_bridge/passes/tao_bace_reformat_pass.h"
#include "tao_bridge/passes/tao_build_tao_op_pass.h"
#include "tao_bridge/passes/tao_clone_constants_for_better_clustering.h"
#include "tao_bridge/passes/tao_defuse_pass.h"
#include "tao_bridge/passes/tao_encapsulate_subgraphs_pass.h"
#include "tao_bridge/passes/tao_mark_for_compilation_pass.h"
#include "tao_bridge/passes/tao_partially_decluster_pass.h"
#include "tao_bridge/tao_util.h"
#include "tao_bridge/tf/dump_graph.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

namespace {

void DumpGraph(const GraphOptimizationPassOptions& options,
               const char* const name) {
  if (GetTaoBridgeOptions()->dump_pass_output || VLOG_IS_ON(2)) {
    Graph* graph = options.graph->get();
    auto dumped = dump_graph::DumpGraphToFile(name, *graph, options.flib_def);
    VLOG(2) << "TaoOptimizationPass dump graph: " << dumped;
  }
}

}  // namespace

Status TaoOptimizationPass::Run(const GraphOptimizationPassOptions& options) {
  static std::atomic<int> optimization_counter{0};

  bool enable_tao = GetTaoBridgeOptions()->enable_tao;
  if (!enable_tao) {
    return Status::OK();
  }

  int optimization_pass_start = GetTaoBridgeOptions()->optimization_pass_start;
  int cur_optimization_counter = optimization_counter++;
  if (cur_optimization_counter < optimization_pass_start) {
    VLOG(0) << "Skip clustering: " << cur_optimization_counter
            << " Start clustering: " << optimization_pass_start;
    return Status::OK();
  }

  if (util::HasOpType(**options.graph, "TaoLaunch") ||
      util::HasOpType(**options.graph, "TaoMlirLaunch") ||
      util::HasOpType(**options.graph, "DiscLaunch")) {
    LOG(INFO) << "Skip optimize graph which contains TaoLaunch op.";
    return Status::OK();
  }

  opts_->feats = absl::make_unique<FeatureDetector>("TaoOptimizationPass",
                                                    options.graph->get());
  if (opts_->feats->HasGradientOp()) {
    if (GetTaoBridgeOptions()->skip_training_graph &&
        !FeatureDetector::IsDistributedForceOn()) {
      LOG(INFO)
          << "Skip optimizing graph which contains gradient op and not in "
             "dist white list";
      return Status::OK();
    }
    // set default parameter for training tasks
    // only work for XLA. not work when MLIR is enabled.
    // will override tf_xla_min_cluster_size & tf_xla_min_cluster_size set by
    // TF_XLA_FLAGS in TaoMarkForCompilationPass
    auto val = GetTaoBridgeOptions()->train_task_max_cluster_size;
    if (val != -1) {
      opts_->max_cluster_size = val;
    }
    val = GetTaoBridgeOptions()->train_task_min_cluster_size;
    if (val != -1) {
      opts_->min_cluster_size = val;
    }
  }

  DumpGraph(options, "before_bace_pass");

  if (GetTaoBridgeOptions()->tao_mlir_branch_only) {
    auto tao_passes_opt = absl::make_unique<TaoPassOptions>();
    tao_passes_opt->override_tf_xla_ops_to_cluster = "MLIR";
    // '0' means use default value or global setting.
    tao_passes_opt->min_cluster_size = 0;
    tao_passes_opt->inner_tao_launch = true;
    tao_passes_opt->cluster_recount = false;
    this->set_options(std::move(tao_passes_opt));
  }

  TaoBaCEReformatPass bace_pass;
  bace_pass.set_name("TaoBaCEReformatPass");
  TF_RETURN_IF_ERROR(bace_pass.Run(options));

  DumpGraph(options, "before_tao_clone_pass");

  bool mlir_whole_graph_compilation =
      GetTaoBridgeOptions()->experimental_enable_mlir_whole_graph_compilation;
  auto flags = GetMarkForCompilationPassFlags();
  if (flags->tf_xla_cpu_global_jit) {
    // DumpGraph(options, "before_tao_defuse_pass");
    // Add Defuse Pass for CPU case to handle fusion from grappler 1.15
    // Be moved into TF2XLA
    TaoDefusePass defuse_pass;
    defuse_pass.set_name("TaoDefusePass");
    TF_RETURN_IF_ERROR(defuse_pass.Run(options));
  }

  DumpGraph(options, "before_tao_clone_pass");
  // Add Clone Constants Pass for fakequant
  TaoCloneConstantsForBetterClusteringPass clone_pass;
  clone_pass.set_name("TaoCloneConstantsForFakeQuantPass");
  TF_RETURN_IF_ERROR(clone_pass.Run(options));

  DumpGraph(options, "before_tao_mark_pass");

  TaoMarkForCompilationPass mark_pass;
  mark_pass.set_name("TaoMarkForCompilationPass");
  mark_pass.set_opts(opts_);
  TF_RETURN_IF_ERROR(mark_pass.Run(options));

  if (!mlir_whole_graph_compilation) {
    DumpGraph(options, "before_tao_decluster_pass");

    TaoPartiallyDeclusterPass decluster_pass;
    decluster_pass.set_name("TaoPartiallyDeclusterPass");
    TF_RETURN_IF_ERROR(decluster_pass.Run(options));
    decluster_pass.set_opts(opts_);
  }

  DumpGraph(options, "before_tao_encap_pass");

  TaoEncapsulateSubgraphsPass encap_pass;
  encap_pass.set_name("TaoEncapsulateSubgraphsPass");
  encap_pass.set_opts(opts_);
  TF_RETURN_IF_ERROR(encap_pass.Run(options));

  DumpGraph(options, "before_tao_build_op_pass");

  TaoBuildTaoOpPass op_pass;
  op_pass.set_name("TaoBuildTaoOpPass");
  op_pass.set_opts(opts_);
  TF_RETURN_IF_ERROR(op_pass.Run(options));

  DumpGraph(options, "after_tao_pass");

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      TaoOptimizationPass);

}  // namespace tao
}  // namespace tensorflow
