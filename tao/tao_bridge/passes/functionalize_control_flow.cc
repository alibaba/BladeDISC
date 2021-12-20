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

// Adopted from tensorflow/compiler/tf2xla/functionalize_control_flow.cc

#include "tao_bridge/passes/functionalize_control_flow.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tao_bridge/passes/defunctionalize_control_flow.h"
#include "tao_bridge/passes/functionalize_cond.h"
#include "tao_bridge/passes/functionalize_control_flow_util.h"
#include "tao_bridge/passes/functionalize_while.h"
#include "tao_bridge/passes/tao_mark_for_compilation_pass.h"
#include "tao_bridge/tf/dump_graph.h"
#include "tao_bridge/tf/statusor.h"
#include "tao_bridge/tf/tf2xla_util.h"
#include "tao_bridge/tf/union_find.h"
#include "tao_bridge/tf/xla_op_registry.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

namespace {

void DumpGraph(const GraphOptimizationPassOptions& options,
               const char* const name) {
  if (GetTaoBridgeOptions()->dump_pass_output || VLOG_IS_ON(2)) {
    Graph* graph = options.graph->get();
    auto dumped = dump_graph::DumpGraphToFile(name, *graph, options.flib_def);
    VLOG(2) << "FunctionalizeControlFlowPass dump graph: " << dumped;
  }
}

Status MaybeDefunctionalize(Graph* graph, FunctionLibraryDefinition* library,
                            FunctionLibraryRuntime* flr) {
  // Register ops as need to check whether compilable
  XlaOpRegistry::RegisterCompilationKernels();

  RecursiveCompilabilityChecker::OperationFilter op_filter =
      CreateAllowAllOperationFilter();
  DeviceType jit_device_type(DEVICE_TAO_GPU_XLA_JIT);
  RecursiveCompilabilityChecker jit_recursive_checker(&op_filter,
                                                      &jit_device_type);
  // NOTE(pengzhan): the GPU checker is used to detect ops which don't have
  // registered GPU kernels. Some kernels may have registered by compiler but
  // not by host TF. This ensures the fallback graph can be executed
  // successfully, but leave space for improvement.
  DeviceType device_type(DEVICE_GPU);
  RecursiveCompilabilityChecker recursive_checker(&op_filter, &device_type);

  // This ensures that nested If/While nodes will be lowered as well.
  for (int i = 2; i < graph->num_node_ids(); ++i) {
    Node* node = graph->FindNodeId(i);
    if (node == nullptr) continue;  // deleted node
    if (IsFunctionalControlFlowOps(node) &&
        (!jit_recursive_checker.IsCompilableFunctionalOp(*node, flr) ||
         !recursive_checker.IsCompilableFunctionalOp(*node, flr))) {
      // TODO(pengzhan): make sure there is no need to do recursive lower
      VLOG(2) << node->name() << " is defunctionalized";
      TF_RETURN_IF_ERROR(
          DefunctionalizeFactory().defunctionalize(node, graph, *library));
    }
  }
  return Status::OK();
}

Status CheckSoftPlacement(const Graph* graph, bool allow_soft_placement) {
  // We should throw an error when not set allow_soft_placement, but force int32
  // TensorArray ops on GPU, as there is no registered kernel in host TF. We can
  // not guarantee int32 TensorArray ops will be successfully compiled.
  if (allow_soft_placement) {
    return Status::OK();
  }
  for (const Node* node : graph->op_nodes()) {
    if (IsInvalidTensorArrayOps(node) && !node->requested_device().empty()) {
      DeviceType device_type("");
      TF_RETURN_IF_ERROR(
          DeviceNameToDeviceType(node->requested_device(), &device_type));
      if (device_type == DEVICE_GPU) {
        return errors::Internal("Requested device '", node->requested_device(),
                                "' does not have registered OpKernel support "
                                "for ",
                                node->type_string());
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status FunctionalizeControlFlow(const FunctionLibraryDefinition* lookup_library,
                                Graph* graph,
                                FunctionLibraryDefinition* library,
                                FunctionLibraryRuntime* flr) {
  // Functionalize and remove while loops from graph.
  TF_RETURN_IF_ERROR(FunctionalizeWhileLoop(lookup_library, graph, library));
  // FunctionalizeControlFlow is invoked for every function, so the loops's
  // bodies and conditionals that were extracted into functions will be handled
  // in successive invocations.
  TF_RETURN_IF_ERROR(FunctionalizeCond(graph, library));

  return Status::OK();
}

Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library,
                                FunctionLibraryRuntime* flr) {
  return FunctionalizeControlFlow(/*lookup_library=*/nullptr, graph, library,
                                  flr);
}

Status FunctionalizeControlFlowPass::Run(
    const GraphOptimizationPassOptions& options) {
  bool enable_tao = GetTaoBridgeOptions()->enable_tao;
  bool enable_control_flow = GetTaoBridgeOptions()->tao_enable_control_flow;
  if (enable_tao && enable_control_flow) {
    VLOG(0) << "Enable FunctionalizeControlFlowPass";

    Graph* graph = options.graph->get();
    DumpGraph(options, "before_functionalize_control_flow");

    TF_RETURN_IF_ERROR(CheckSoftPlacement(
        graph, options.session_options != nullptr &&
                   options.session_options->config.allow_soft_placement()));

    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
#if TF_MAJOR_VERSION > 1
        // TF2.4
        new ProcessFunctionLibraryRuntime(
            /*device_mgr=*/nullptr, options.session_options->env,
            /*new: config=*/&options.session_options->config,
            TF_GRAPH_DEF_VERSION, options.flib_def, OptimizerOptions()));
#else
        // TF1.12, TF1.15
        new ProcessFunctionLibraryRuntime(
            /*device_mgr=*/nullptr, options.session_options->env,
            TF_GRAPH_DEF_VERSION, options.flib_def, OptimizerOptions()));
#endif
    FunctionLibraryRuntime* flr =
        pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

    CHECK(flr != nullptr);
    TF_RETURN_IF_ERROR(FunctionalizeControlFlow(graph, options.flib_def, flr));

    DumpGraph(options, "after_functionalize_control_flow");

    TF_RETURN_IF_ERROR(MaybeDefunctionalize(graph, options.flib_def, flr));

    DumpGraph(options, "after_defunctionalize");
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 27,
                      FunctionalizeControlFlowPass);

}  // namespace tao
}  // namespace tensorflow
