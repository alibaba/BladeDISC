#include "tao_bridge/passes/tao_remove_small_cluster_pass.h"

#include <sstream>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"
#include "tao_bridge/test_helpers.h"
#include "tao_bridge/tf/xla_cluster_util.h"
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

Status RemoveCluster(std::unique_ptr<Graph>* graph) {
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
  TaoRemoveSmallClusterPass pass(opts->use_tvm);

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

TEST(RemoveClusterPassTest, Base) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* data = ops::SourceOp("FakeNullary", builder.opts().WithName("data"));
    Node* filter =
        ops::SourceOp("FakeNullary", builder.opts().WithName("filter"));
    Node* constant = ops::SourceOp("Const", builder.opts()
                                                .WithName("Const")
                                                .WithAttr("dtype", DT_FLOAT)
                                                .WithAttr("value", Tensor()));

    Node* op_0 = ops::BinaryOp("FakeBinary", data, constant,
                               builder.opts().WithName("Operation_0"));
    Node* op_1 = ops::BinaryOp("Conv2D", data, filter,
                               builder.opts()
                                   .WithName("Operation_1")
                                   .WithAttr("strides", {1, 1, 1, 1})
                                   .WithAttr("padding", "VALID")
                                   .WithAttr("explicit_paddings", {0, 0, 0, 0})
                                   .WithAttr("data_format", "NHWC")
                                   .WithAttr("dilations", {1, 1, 1, 1})
                                   .WithAttr("T", DT_FLOAT));
    op_0->AddAttr(kXlaClusterAttr, "cluster_0");
    op_1->AddAttr(kXlaClusterAttr, "cluster_1");

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  Node* op_0_node = FindNodeByName(*graph, "Operation_0");
  Node* op_1_node = FindNodeByName(*graph, "Operation_1");

  ASSERT_EQ(GetXlaClusterForNode(*op_0_node), "cluster_0");
  ASSERT_EQ(GetXlaClusterForNode(*op_1_node), "cluster_1");

  TF_ASSERT_OK(RemoveCluster(&graph));

  op_0_node = FindNodeByName(*graph, "Operation_0");
  op_1_node = FindNodeByName(*graph, "Operation_1");

  TaoPassOptions opts;
  if (opts.use_tvm) {
    ASSERT_EQ(GetXlaClusterForNode(*op_0_node), absl::nullopt);
    ASSERT_EQ(GetXlaClusterForNode(*op_1_node), "cluster_1");
  } else {
    ASSERT_EQ(GetXlaClusterForNode(*op_0_node), "cluster_0");
    ASSERT_EQ(GetXlaClusterForNode(*op_1_node), "cluster_1");
  }
}

}  // namespace
}  // namespace tao
}  // namespace tensorflow
