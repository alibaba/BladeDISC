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

#include <sstream>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"
#include "tao_bridge/test_helpers.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
namespace tao {

namespace {
std::unique_ptr<Graph> CreateGraph() {
  std::stringstream ss;
  // clang-format off
  ss << "node {\n\
  op: 'FakeNullary'\n\
  name: 'data'\n\
}\n\
node {\n\
  op: 'FakeNullary'\n\
  name: 'weight'\n\
}\n\
node {\n\
  op: 'FakeNullary'\n\
  name: 'gamma'\n\
}\n\
node {\n\
  op: 'FakeNullary'\n\
  name: 'beta'\n\
}\n\
node {\n\
  op: 'FakeNullary'\n\
  name: 'mean'\n\
}\n\
node {\n\
  op: 'FakeNullary'\n\
  name: 'variance'\n\
}\n\
node {\n\
  name: 'fusedconv2d'\n\
  op: '_FusedConv2D'\n\
  input: 'data'\n\
  input: 'weight'\n\
  input: 'gamma'\n\
  input: 'beta'\n\
  input: 'mean'\n\
  input: 'variance'\n\
  device: '/job:localhost/replica:0/task:0/device:CPU:0'\n\
  attr {\n\
    key: 'T'\n\
    value {\n\
      type: DT_FLOAT\n\
    }\n\
  }\n\
  attr {\n\
    key: 'data_format'\n\
    value {\n\
      s: 'NHWC'\n\
    }\n\
  }\n\
  attr {\n\
    key: 'dilations'\n\
    value {\n\
      list {\n\
        i: 1\n\
        i: 1\n\
        i: 1\n\
        i: 1\n\
      }\n\
    }\n\
  }\n\
  attr {\n\
    key: 'epsilon'\n\
    value {\n\
      f: 2e-05\n\
    }\n\
  }\n\
  attr {\n\
    key: 'explicit_paddings'\n\
    value {\n\
      list {\n\
      }\n\
    }\n\
  }\n\
  attr {\n\
    key: 'fused_ops'\n\
    value {\n\
      list {\n\
        s: 'FusedBatchNorm'\n\
        s: 'Relu'\n\
      }\n\
    }\n\
  }\n\
  attr {\n\
    key: 'num_args'\n\
    value {\n\
      i: 4\n\
    }\n\
  }\n\
  attr {\n\
    key: 'padding'\n\
    value {\n\
      s: 'VALID'\n\
    }\n\
  }\n\
  attr {\n\
    key: 'strides'\n\
    value {\n\
      list {\n\
        i: 1\n\
        i: 2\n\
        i: 2\n\
        i: 1\n\
      }\n\
    }\n\
  }\n\
  attr {\n\
    key: 'use_cudnn_on_gpu'\n\
    value {\n\
      b: true\n\
    }\n\
  }\n\
}";
  // clang-format on
  GraphDef gdef;
  CHECK(protobuf::TextFormat::ParseFromString(ss.str(), &gdef));
  GraphConstructorOptions opts;
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef, g.get()));
  return std::move(g);
}

REGISTER_OP("_FusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("padding: string")
    .Attr("explicit_paddings: list(int)")
    .Attr("data_format: string")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("fused_ops: list(string) = []")
    .Attr("epsilon: float = 0.0001");

REGISTER_OP("Conv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("padding: string")
    .Attr("explicit_paddings: list(int)")
    .Attr("data_format: string")
    .Attr("dilations: list(int) = [1, 1, 1, 1]");

REGISTER_OP("FusedBatchNorm")
    .Input("x: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Output("y: T")
    .Attr("T: {float}")
    .Attr("data_format: string")
    .Attr("epsilon: float = 0.0001")
    .Attr("is_training: bool = true");

REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {realnumbertype}");

REGISTER_OP("Relu6")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {realnumbertype}");

REGISTER_OP("Elu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {realnumbertype}");

Status Defuse(std::unique_ptr<Graph>* graph) {
  FixupSourceAndSinkEdges(graph->get());
  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kCpuDevice);
    }
  }
  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;

  std::unique_ptr<TaoPassOptions> opts = absl::make_unique<TaoPassOptions>();
  TaoDefusePass pass;
  // pass.set_opts(opts);
  return pass.Run(opt_options);
}

Node* FindNodeByName(const Graph& graph, const string& name) {
  for (Node* node : graph.nodes()) {
    if (node->name() == name) {
      return node;
    }
  }
  return nullptr;
}

bool GetInputsForNode(const Graph& graph, const string& node_name,
                      std::vector<Node*>* inputs) {
  const Node* node = FindNodeByName(graph, node_name);
  if (node == nullptr) {
    return false;
  }
  for (const Edge* e : node->in_edges()) {
    inputs->push_back(e->src());
  }
  std::sort(inputs->begin(), inputs->end(), NodeComparatorName());
  return true;
}

TEST(TaoDefusePassTest, Base) {
  auto g = CreateGraph();

  TF_ASSERT_OK(Defuse(&g));
  int node_cnt = 0;
  for (auto* node : g->op_nodes()) {
    node_cnt++;
  }
  Node* conv_node = FindNodeByName(*g, "fusedconv2d_conv_2d");
  Node* bn_node = FindNodeByName(*g, "fusedconv2d_batchnorm");
  Node* relu_node = FindNodeByName(*g, "fusedconv2d_Relu");

  TaoPassOptions opts;
  EXPECT_EQ(node_cnt, 9);
  EXPECT_NE(conv_node, nullptr);
  EXPECT_NE(bn_node, nullptr);
  EXPECT_NE(relu_node, nullptr);
}
}  // namespace
}  // namespace tao
}  // namespace tensorflow
