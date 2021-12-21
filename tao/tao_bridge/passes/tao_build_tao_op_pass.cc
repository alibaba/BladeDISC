#include "tao_bridge/passes/tao_build_tao_op_pass.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tao_bridge/passes/tao_encapsulate_subgraphs_pass.h"
#include "tao_bridge/tao_util.h"
#include "tao_bridge/tf/dump_graph.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tao {

namespace {

struct XlaClusterInfo {
  std::vector<NodeBuilder::NodeOut> constant_inputs;
  std::vector<NodeBuilder::NodeOut> fixed_shape_inputs;
  std::vector<NodeBuilder::NodeOut> non_const_or_fixedshape_host_inputs;
  std::vector<NodeBuilder::NodeOut> non_const_or_fixedshape_device_inputs;
  std::vector<NodeBuilder::NodeOut> resource_inputs;
  DataTypeVector host_rets;
  DataTypeVector device_rets;
  NameAttrList function;
};

NodeBuilder::NodeOut IncomingEdgeAsOutput(const Edge* e) {
  return NodeBuilder::NodeOut(e->src(), e->src_output());
}

Status GetXlaClusterInfo(Node* n, XlaClusterInfo* result) {
  int num_constant_inputs = 0;
  int num_fixed_shape_inputs = 0;
  int num_non_const_or_fixedshape_host_inputs = 0;
  int num_resource_inputs = 0;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumConstantArgsAttr, &num_constant_inputs));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kMlirNumFixedShapeArgsAttr,
                                 &num_fixed_shape_inputs));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kMlirNumHostArgsAttr,
                                 &num_non_const_or_fixedshape_host_inputs));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumResourceArgsAttr, &num_resource_inputs));

  int num_non_const_or_fixedshape_device_inputs =
      n->num_inputs() - num_constant_inputs - num_fixed_shape_inputs -
      num_non_const_or_fixedshape_host_inputs - num_resource_inputs;

  if (num_constant_inputs < 0 || num_resource_inputs < 0 ||
      num_non_const_or_fixedshape_host_inputs < 0 ||
      num_non_const_or_fixedshape_device_inputs < 0) {
    return errors::InvalidArgument(
        "Invalid number of constant/fixedshape/resource arguments to XLA "
        "kernel.");
  }

  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  int idx = 0;
  for (const Edge* e : input_edges_vector) {
    if (idx < num_constant_inputs) {
      result->constant_inputs.push_back(IncomingEdgeAsOutput(e));
    } else if (idx < num_constant_inputs + num_fixed_shape_inputs) {
      result->fixed_shape_inputs.push_back(IncomingEdgeAsOutput(e));
    } else if (idx < num_constant_inputs + num_fixed_shape_inputs +
                         num_non_const_or_fixedshape_host_inputs) {
      result->non_const_or_fixedshape_host_inputs.push_back(
          IncomingEdgeAsOutput(e));
    } else if (idx < num_constant_inputs + num_fixed_shape_inputs +
                         num_non_const_or_fixedshape_host_inputs +
                         num_non_const_or_fixedshape_device_inputs) {
      result->non_const_or_fixedshape_device_inputs.push_back(
          IncomingEdgeAsOutput(e));
    } else {
      result->resource_inputs.push_back(IncomingEdgeAsOutput(e));
    }
    ++idx;
  }

  result->function.set_name(n->type_string());
  *result->function.mutable_attr() = n->def().attr();

  int num_host_rets = 0;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kMlirNumHostRetsAttr, &num_host_rets));
  for (int i = 0; i < n->num_outputs(); ++i) {
    auto t = n->output_type(i);
    if (i < num_host_rets) {
      result->host_rets.push_back(t);
    } else {
      result->device_rets.push_back(t);
    }
  }

  return Status::OK();
}

Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to) {
  for (const Edge* e : from->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), to);
    }
  }

  return Status::OK();
}

void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node) {
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    // TODO(sanjoy): This does not update NodeDef inputs.  To be able to update
    // NodeDef inputs we first need to fix encapsulate_subgraphs_pass to fix up
    // the NodeDef inputs to the function call nodes.
    g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
    VLOG(1) << new_node->name() << ":" << edge->src_output() << " -> "
            << edge->dst()->name() << ":" << edge->dst_input();
    g->RemoveEdge(edge);
  }
}

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

Status PrepareMLIRClustering(Graph* graph, const std::string& device) {
  for (Node* n : graph->nodes()) {
    n->ClearAttr("_XlaAlreadyClustered");
    if (n->type_string() == "_Arg" || n->type_string() == "_Retval") {
      DataType dtype = DT_INVALID;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
      if (dtype == DT_INT32 || DataTypeAlwaysOnHost(dtype)) {
        n->set_assigned_device_name("/device:CPU:0");
      } else {
        n->set_assigned_device_name(device);
      }
    } else {
      n->set_assigned_device_name(device);
    }
  }
  return Status::OK();
}

/// Return true if the input mlir graph is valid, return false otherwise.
bool ValidateMLIRGraph(const Graph& graph,
                       const FunctionLibraryDefinition* const fld) {
  bool has_tao_launch = false;
  for (auto* node : graph.op_nodes()) {
    if (node->type_string() == "TaoMlirLaunch" ||
        node->type_string() == "DiscLaunch") {
      has_tao_launch = true;

      const AttrValue* attr_value = node->attrs().Find("mlir_function");
      if (!attr_value) {
        VLOG(0) << "Invalid MLIR branch: TaoLaunch op has not `mlir_function` "
                   "attr.";
        return false;
      }
      const std::string& func_name = attr_value->func().name();
      if (func_name.empty()) {
        VLOG(0) << "Invalid MLIR branch: TaoLaunch op has empty "
                   "`mlir_function` attr.";
        return false;
      }
      auto* func_def = fld->Find(func_name);
      if (func_def == nullptr) {
        VLOG(0) << "Invalid MLIR branch: mlir function not found in function "
                   "library definition: "
                << func_name;
        return false;
      }
    }
  }
  if (!has_tao_launch && !GetTaoBridgeOptions()->tao_mlir_branch_only) {
    VLOG(0) << "Invalid MLIR branch: no TaoLaunch op created.";
    return false;
  }
  return true;
}

Status CreateMlirFunction(const GraphOptimizationPassOptions& options, Graph* g,
                          Node* n, const NameAttrList& tf_func,
                          std::unique_ptr<NameAttrList>* mlir_func) {
  // const FunctionLibraryDefinition& flib_def = g->flib_def();
  const FunctionDef* tf_fdef = options.flib_def->Find(tf_func.name());
  if (tf_fdef == nullptr) {
    return errors::Internal("FunctionDef for TaoOp not found:", tf_func.name());
  }

  OptimizerOptions opts;
#if TF_MAJOR_VERSION > 1
  // TF2.4
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env,
      &options.session_options->config /*new*/, TF_GRAPH_DEF_VERSION,
      options.flib_def, opts);
#else
  // TF1.12, TF1.15
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, TF_GRAPH_DEF_VERSION,
      options.flib_def, opts);
#endif

  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  FunctionLibraryRuntime::Handle mlir_func_handle;
  TF_RETURN_IF_ERROR(flr->Instantiate(
      tf_func.name(), AttrSlice(&tf_func.attr()), &mlir_func_handle));
  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(flr->ReleaseHandle(mlir_func_handle)); });
  const FunctionBody* fbody = flr->GetFunctionBody(mlir_func_handle);
  std::unique_ptr<Graph> copy(new Graph(fbody->graph->flib_def()));
  CopyGraph(*fbody->graph, copy.get());
#ifndef TF_1_12
  PrepareMLIRClustering(copy.get(), n->requested_device());
#else
  VLOG(0) << "assigned_device_name = " << n->assigned_device_name();
  PrepareMLIRClustering(copy.get(), n->assigned_device_name());
#endif

  GraphOptimizationPassOptions sub_options;
  sub_options.session_options = options.session_options;
  sub_options.cost_model = options.cost_model;
  sub_options.flib_def = options.flib_def;
  sub_options.device_set = options.device_set;
  sub_options.graph = &copy;
  auto tao_passes_opt = absl::make_unique<TaoPassOptions>();
  tao_passes_opt->min_cluster_size = 1;
  tao_passes_opt->override_tf_xla_ops_to_cluster = "MLIR";
  tao_passes_opt->inner_tao_launch = true;
  tao_passes_opt->cluster_recount = false;
  TaoOptimizationPass pass;
  pass.set_options(std::move(tao_passes_opt));
  TF_RETURN_IF_ERROR(pass.Run(sub_options));

  // check if TaoLaunch op create in mlir function.
  if (!ValidateMLIRGraph(*copy, sub_options.flib_def)) {
    return Status::OK();
  }

  VLOG(1) << "TaoLaunch for MLIR created.";
  *mlir_func = absl::make_unique<NameAttrList>(tf_func);
  std::string mlir_name = tf_func.name() + "_mlir";
  (*mlir_func)->set_name(mlir_name);
  FunctionDef mlir_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*copy, mlir_name, &mlir_fdef));
  if (options.flib_def->Find(mlir_name) != nullptr) {
    return errors::Internal("FunctionDef already exists:", mlir_name);
  }
  TF_RETURN_IF_ERROR(options.flib_def->AddFunctionDef(mlir_fdef));
  return Status::OK();
}

Status CreateFallbackFunction(const GraphOptimizationPassOptions& options,
                              const Node* node, const NameAttrList& tf_func,
                              NameAttrList* fallback_function) {
  FunctionLibraryDefinition* const library = options.flib_def;
  string defunct_name = tf_func.name() + kDefunctionalizedSuffix;
  if (library->Find(defunct_name) != nullptr) {
    fallback_function->set_name(defunct_name);
    *fallback_function->mutable_attr() = node->def().attr();
  }
  return Status::OK();
}

Status ReplaceNodeWithTaoLaunchOp(const GraphOptimizationPassOptions& options,
                                  Graph* g, Node* n, bool inner) {
  VLOG(1) << "Run ReplaceNodeWithTaoLaunchOp with " << n->name();
  XlaClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetXlaClusterInfo(n, &cluster_info));

  auto mlir_func = absl::make_unique<NameAttrList>();
  // Do this at top level clustering.
  if (!inner && GetTaoBridgeOptions()->enable_mlir) {
    TF_RETURN_IF_ERROR(
        CreateMlirFunction(options, g, n, cluster_info.function, &mlir_func));
  }

  NameAttrList fallback_function;
  CreateFallbackFunction(options, n, cluster_info.function, &fallback_function);

  VLOG(1) << "is_mlir: " << inner;
  VLOG(1) << "const_input size: " << cluster_info.constant_inputs.size();
  VLOG(1) << "fixed_shape_input size: "
          << cluster_info.fixed_shape_inputs.size();
  VLOG(1) << "host args size: "
          << cluster_info.non_const_or_fixedshape_host_inputs.size();
  VLOG(1) << "host rets size: " << cluster_info.host_rets.size();
#ifdef PLATFORM_ALIBABA
  NodeBuilder nb =
      inner ? NodeBuilder(n->name() + "_tao_mlir_launch", "TaoMlirLaunch")
                  .Input(cluster_info.constant_inputs)
                  .Input(cluster_info.fixed_shape_inputs)
                  .Input(cluster_info.non_const_or_fixedshape_host_inputs)
                  .Input(cluster_info.non_const_or_fixedshape_device_inputs)
                  .Input(cluster_info.resource_inputs)
                  .Attr("Thostresults", cluster_info.host_rets)
                  .Attr("Tdeviceresults", cluster_info.device_rets)
                  .Attr("_XlaCompile", false)
                  .Attr("function", NameAttrList())
                  .Attr("fallback_function", NameAttrList())
                  // When not at top level clustering, mlir_func is
                  // always nullptr, the func_def is extracted by
                  // GetXlaClusterInfo and it's mlir function!
                  .Attr("mlir_function", cluster_info.function)
                  .Attr("inner", inner)
                  .Device(n->requested_device())
                  .AssignedDevice(n->assigned_device_name())
            : NodeBuilder(n->name() + "_tao_launch", "TaoLaunch")
                  .Input(cluster_info.constant_inputs)
                  .Input(cluster_info.non_const_or_fixedshape_device_inputs)
                  .Input(cluster_info.resource_inputs)
                  .Attr("Tresults", n->output_types())
                  .Attr("_XlaCompile", false)
                  .Attr("function", cluster_info.function)
                  .Attr("fallback_function", fallback_function)
                  .Attr("mlir_function", *mlir_func)
                  .Attr("inner", inner)
                  .Device(n->requested_device())
                  .AssignedDevice(n->assigned_device_name());
#else
  NodeBuilder nb =
      NodeBuilder(n->name() + "_disc_launch", "DiscLaunch")
          .Input(cluster_info.constant_inputs)
          .Input(cluster_info.fixed_shape_inputs)
          .Input(cluster_info.non_const_or_fixedshape_host_inputs)
          .Input(cluster_info.non_const_or_fixedshape_device_inputs)
          .Input(cluster_info.resource_inputs)
          .Attr("Thostresults", cluster_info.host_rets)
          .Attr("Tdeviceresults", cluster_info.device_rets)
          .Attr("_XlaCompile", false)
          // When not at top level clustering, mlir_func is
          // always nullptr, the func_def is extracted by
          // GetXlaClusterInfo and it's mlir function!
          .Attr("mlir_function", cluster_info.function)
          .Device(n->requested_device())
          .AssignedDevice(n->assigned_device_name());
#endif
  Node* tao_op = nullptr;
  Status status = nb.Finalize(g, &tao_op);
  TF_CHECK_OK(status);

  TF_RETURN_IF_ERROR(CopyIncomingControlEdges(g, /*from=*/n, /*to=*/tao_op));

  MoveOutgoingEdges(g, /*old_node=*/n, /*new_node=*/tao_op);
  g->RemoveNode(n);
  return Status::OK();
}

}  // namespace

Status TaoBuildTaoOpPass::Run(const GraphOptimizationPassOptions& options) {
  VLOG(1) << "TaoBuildTaoOpPass::Run is called, inner TaoLaunch: "
          << inner_tao_launch_;
  Graph* graph = options.graph->get();

  std::vector<Node*> target_nodes;
  for (Node* n : graph->op_nodes()) {
    // In all cases, only try to compile computational nodes.
    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
      continue;
    }
    if (IsXlaCompiledKernel(*n)) {
      target_nodes.push_back(n);
    }
  }
  for (auto n : target_nodes) {
    TF_RETURN_IF_ERROR(
        ReplaceNodeWithTaoLaunchOp(options, graph, n, inner_tao_launch_));
  }

  if (false) {
    VLOG(0) << dump_graph::DumpGraphToFile("build_tao_ops after", *graph,
                                           options.flib_def);
  }
  return Status::OK();
}

}  // namespace tao
}  // namespace tensorflow
