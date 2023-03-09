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

#include "tao_bridge/passes/tao_defuse_pass.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tao_bridge/passes/tao_encapsulate_subgraphs_pass.h"
#include "tao_bridge/tao_util.h"
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

bool IsActivation(string act) {
  if (act == "Relu" || act == "Relu6" || act == "Elu") {
    return true;
  }
  return false;
}

NodeBuilder::NodeOut IncomingEdgeAsOutput(const Edge* e) {
  return NodeBuilder::NodeOut(e->src(), e->src_output());
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

StatusOr<Node*> DefuseMatMul(Graph* g, Node* n,
                             std::vector<NodeBuilder::NodeOut>& in_edges) {
  DataType datatype;
  bool transpose_a;
  bool transpose_b;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &datatype));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "transpose_b", &transpose_b));

  NodeBuilder nb = NodeBuilder(n->name() + "_malmul", "MatMul")
                       .Input(in_edges[0])
                       .Input(in_edges[1])
                       .Attr("T", datatype)
                       .Attr("transpose_a", transpose_a)
                       .Attr("transpose_b", transpose_b)
                       .Device(n->requested_device())
                       .AssignedDevice(n->assigned_device_name());

  Node* mm_op = nullptr;
  Status status = nb.Finalize(g, &mm_op);
  TF_CHECK_OK(status);

  return mm_op;
}

StatusOr<Node*> DefuseBiasAdd(Graph* g, Node* n,
                              std::vector<NodeBuilder::NodeOut>& in_edges) {
  DataType datatype;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &datatype));

  NodeBuilder nb = NodeBuilder(n->name() + "_biasadd", "BiasAdd")
                       .Input(in_edges[0])
                       .Input(in_edges[1])
                       .Attr("T", datatype)
                       .Device(n->requested_device())
                       .AssignedDevice(n->assigned_device_name());

  Node* ba_op = nullptr;
  Status status = nb.Finalize(g, &ba_op);
  TF_CHECK_OK(status);

  return ba_op;
}

StatusOr<Node*> DefuseActivation(Graph* g, Node* n,
                                 std::vector<NodeBuilder::NodeOut>& in_edges,
                                 const string& name) {
  DataType datatype;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &datatype));

  NodeBuilder nb = NodeBuilder(n->name() + "_" + name, name)
                       .Input(in_edges[0])
                       .Attr("T", datatype)
                       .Device(n->requested_device())
                       .AssignedDevice(n->assigned_device_name());

  Node* act_op = nullptr;
  Status status = nb.Finalize(g, &act_op);
  TF_CHECK_OK(status);

  return act_op;
}

StatusOr<Node*> DefuseBatchNorm(Graph* g, Node* n,
                                std::vector<NodeBuilder::NodeOut>& in_edges) {
  DataType datatype;
  string data_format;
  float epsilon;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &datatype));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "epsilon", &epsilon));

  NodeBuilder nb = NodeBuilder(n->name() + "_batchnorm", "FusedBatchNorm")
                       .Input(in_edges[0])
                       .Input(in_edges[1])
                       .Input(in_edges[2])
                       .Input(in_edges[3])
                       .Input(in_edges[4])
                       .Attr("epsilon", epsilon)
                       .Attr("data_format", data_format)
                       .Attr("is_training", false)
                       .Attr("T", datatype)
                       .Device(n->requested_device())
                       .AssignedDevice(n->assigned_device_name());

  Node* bn_op = nullptr;
  Status status = nb.Finalize(g, &bn_op);
  TF_CHECK_OK(status);

  return bn_op;
}

StatusOr<Node*> DefuseConv2D(Graph* g, Node* n,
                             std::vector<NodeBuilder::NodeOut>& in_edges) {
  DataType datatype;
  std::vector<int> dilations;
  std::vector<int> strides;
  std::vector<int> paddings;
  string padding;
  string data_format;
  bool use_cudnn;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "explicit_paddings", &paddings));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "dilations", &dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "use_cudnn_on_gpu", &use_cudnn));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &datatype));

  NodeBuilder nb = NodeBuilder(n->name() + "_conv_2d", "Conv2D")
                       .Input(in_edges[0])
                       .Input(in_edges[1])
                       .Attr("dilations", dilations)
                       .Attr("use_cudnn_on_gpu", use_cudnn)
                       .Attr("data_format", data_format)
                       .Attr("strides", strides)
                       .Attr("explicit_paddings", paddings)
                       .Attr("padding", padding)
                       .Attr("T", datatype)
                       .Device(n->requested_device())
                       .AssignedDevice(n->assigned_device_name());

  Node* conv_op = nullptr;
  Status status = nb.Finalize(g, &conv_op);
  TF_CHECK_OK(status);
  return conv_op;
}

Status DefuseFromRoot(Graph* g, Node* n) {
  std::vector<string> ops;

  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "fused_ops", &ops));

  std::vector<NodeBuilder::NodeOut> fused_in_edges;
  for (const Edge* e : n->in_edges()) {
    fused_in_edges.push_back(IncomingEdgeAsOutput(e));
  }

  int i = 0;
  Node* op;
  CHECK(n->type_string() != "_FusedBatchNormEx")
      << "No _FusedBatchNormEx for CPU case.";

  if (n->type_string() == "_FusedConv2D") {
    VLOG(2) << "Defuse for _FusedConv2D.";
    std::vector<NodeBuilder::NodeOut> conv_in_edges(fused_in_edges.begin(),
                                                    fused_in_edges.begin() + 2);
    i = i + 2;
    TF_ASSIGN_OR_RETURN(op, DefuseConv2D(g, n, conv_in_edges));
  } else {  // if (n->type_string() == "_FusedMatMul") {
    CHECK(n->type_string() == "_FusedMatMul");
    VLOG(2) << "Defuse for _FusedMatMul.";
    std::vector<NodeBuilder::NodeOut> dense_in_edges(
        fused_in_edges.begin(), fused_in_edges.begin() + 2);
    i = i + 2;
    TF_ASSIGN_OR_RETURN(op, DefuseMatMul(g, n, dense_in_edges));
  }

  TF_RETURN_IF_ERROR(CopyIncomingControlEdges(g, n, op));

  for (auto a : ops) {
    std::vector<NodeBuilder::NodeOut> pre_edges;
    pre_edges.push_back(NodeBuilder::NodeOut(op, 0));
    if (a == "FusedBatchNorm") {
      pre_edges.insert(pre_edges.end(), fused_in_edges.begin() + i,
                       fused_in_edges.begin() + i + 4);
      i = i + 4;
      TF_ASSIGN_OR_RETURN(op, DefuseBatchNorm(g, n, pre_edges));
    }
    if (IsActivation(a)) {
      TF_ASSIGN_OR_RETURN(op, DefuseActivation(g, n, pre_edges, a));
    }
    if (a == "BiasAdd") {
      pre_edges.insert(pre_edges.end(), fused_in_edges.begin() + i,
                       fused_in_edges.begin() + i + 1);
      i = i + 1;
      TF_ASSIGN_OR_RETURN(op, DefuseBiasAdd(g, n, pre_edges));
    }
  }

  VLOG(2) << "Complete defuse for node " << n->name();
  MoveOutgoingEdges(g, n, op);
  g->RemoveNode(n);
  return Status::OK();
}

}  // namespace

Status TaoDefusePass::Run(const GraphOptimizationPassOptions& options) {
  VLOG(2) << "Start defuse pass.";

  Graph* graph = options.graph->get();

  std::vector<Node*> target_nodes;
  for (Node* n : graph->op_nodes()) {
    // In all cases, only try to compile computational nodes.
    if (FusedOpList.count(n->type_string()) > 0) {
      target_nodes.push_back(n);
    }
  }
  VLOG(2) << "Target defuse ops count is " << target_nodes.size();

  for (auto n : target_nodes) {
    TF_RETURN_IF_ERROR(DefuseFromRoot(graph, n));
  }
  return Status::OK();
}

}  // namespace tao
}  // namespace tensorflow
