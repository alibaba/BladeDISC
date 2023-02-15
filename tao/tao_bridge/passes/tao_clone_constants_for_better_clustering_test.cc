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

#include "tao_bridge/passes/tao_clone_constants_for_better_clustering.h"

#include <sstream>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"
#include "tao_bridge/test_helpers.h"
#include "tao_bridge/tf/xla_op_registry.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"

namespace tensorflow {
namespace tao {

namespace {
REGISTER_OP("Const")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type");

REGISTER_OP("NoOp");

REGISTER_OP("FakeNullary").Output("out: float");

REGISTER_OP("FakeBinary")
    .Input("input_0: float")
    .Input("input_1: float")
    .Output("output: float");

class FakeBinaryOp : public OpKernel {
 public:
  explicit FakeBinaryOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override { CHECK(false); }
};

REGISTER_KERNEL_BUILDER(Name("FakeBinary").Device(DEVICE_CPU), FakeBinaryOp);

Status CloneConstant(std::unique_ptr<Graph>* graph) {
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

  TaoCloneConstantsForBetterClusteringPass pass;
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

TEST(CloneConstantPassTest, Base) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input_0 =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input_0"));
    Node* input_1 =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input_1"));
    Node* constant = ops::SourceOp("Const", builder.opts()
                                                .WithName("Const")
                                                .WithAttr("dtype", DT_FLOAT)
                                                .WithAttr("value", Tensor()));

    ops::BinaryOp("FakeBinary", input_0, constant,
                  builder.opts().WithName("Operation_0"));
    ops::BinaryOp("FakeBinary", input_1, constant,
                  builder.opts().WithName("Operation_1"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }
  std::vector<Node*> operation_0_inputs;
  std::vector<Node*> operation_1_inputs;

  ASSERT_TRUE(GetInputsForNode(*graph, "Operation_0", &operation_0_inputs));
  ASSERT_TRUE(GetInputsForNode(*graph, "Operation_1", &operation_1_inputs));
  ASSERT_EQ(operation_0_inputs.size(), 2);
  ASSERT_EQ(operation_1_inputs.size(), 2);
  string name_0, name_1;
  for (auto x : operation_0_inputs) {
    if (x->type_string() == "Const") {
      name_0 = x->name();
    }
  }
  for (auto x : operation_1_inputs) {
    if (x->type_string() == "Const") {
      name_1 = x->name();
    }
  }
  EXPECT_EQ(name_0, name_1);

  TF_ASSERT_OK(CloneConstant(&graph));

  operation_0_inputs.clear();
  operation_1_inputs.clear();
  ASSERT_TRUE(GetInputsForNode(*graph, "Operation_0", &operation_0_inputs));
  ASSERT_TRUE(GetInputsForNode(*graph, "Operation_1", &operation_1_inputs));
  ASSERT_EQ(operation_0_inputs.size(), 2);
  ASSERT_EQ(operation_1_inputs.size(), 2);
  for (auto x : operation_0_inputs) {
    if (x->type_string() == "Const") {
      name_0 = x->name();
    }
  }
  for (auto x : operation_1_inputs) {
    if (x->type_string() == "Const") {
      name_1 = x->name();
    }
  }
  TaoPassOptions opts;
  EXPECT_EQ(name_0, name_1);
}

}  // namespace
}  // namespace tao
}  // namespace tensorflow
