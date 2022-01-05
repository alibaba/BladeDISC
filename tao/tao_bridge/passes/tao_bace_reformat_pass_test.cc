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

#include "tao_bridge/passes/tao_bace_reformat_pass.h"

#include <sstream>

#include "absl/memory/memory.h"
#include "gtest/gtest.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"
#define EIGEN_USE_THREADS
#include "tao_bridge/test_helpers.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace tao {

namespace {
std::unique_ptr<Graph> CreateGraph(absl::string_view dtype) {
  std::stringstream ss;
  // clang-format off
  ss << "node {"
     << "	name: 'A'"
     << "	op: 'Const'"
     << "	attr {"
     << "		key: 'dtype'"
     << "		value {"
     << "			type: " << dtype 
     << "		}"
     << "	}"
     << "	attr {"
     << "		key: 'value'"
     << "		value {"
     << "  			tensor {"
     << "  				dtype: " << dtype
     << "  				tensor_shape {"
     << "  					dim { size: 2 }"
     << "  					dim { size: 2 }"
     << "  				}"
     << "  				float_val: 0"
     << "  				float_val: 1"
     << "  				float_val: 2"
     << "  				float_val: 3"
     << "			}"
     << "		}"
     << "	}"
     << "}";
  // clang-format on
  GraphDef gdef;
  CHECK(protobuf::TextFormat::ParseFromString(ss.str(), &gdef));
  GraphConstructorOptions opts;
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef, g.get()));
  return std::move(g);
}

void RunPass(std::unique_ptr<Graph>* g, int64 max_dim_bar, int64 min_dim_bar,
             int64 size_bar) {
#ifdef TF_1_12
  auto device =
      std::unique_ptr<Device>(new SingleThreadedCpuDevice(Env::Default()));
#else
  auto device =
      std::unique_ptr<Device>(NewSingleThreadedCpuDevice(Env::Default()));
#endif  // TF_1_12

  DeviceSet device_set;
  device_set.AddDevice(device.get());
  device_set.set_client_device(device.get());
  TaoBaCEReformatPass pass(true, max_dim_bar, min_dim_bar, size_bar);
  GraphOptimizationPassOptions opt_options;
  opt_options.device_set = &device_set;
  opt_options.graph = g;
  auto s = pass.Run(opt_options);
  ASSERT_TRUE(pass.Run(opt_options).ok());
}

}  // namespace

REGISTER_OP("Const")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type");

REGISTER_OP("NoOp");

TEST(TaoBaCEReformatPassTest, TestReformat) {
  auto g = CreateGraph("DT_FLOAT");
  RunPass(&g, 1, 1, 1);
  int const_cnt = 0;
  for (auto* node : g->op_nodes()) {
    const_cnt += 1;
    EXPECT_EQ(node->type_string(), "Const");
    EXPECT_EQ(node->attrs().Find("dtype")->type(), DT_HALF);
    auto proto = node->attrs().Find("value")->tensor();
    EXPECT_EQ(proto.dtype(), DT_HALF);
    EXPECT_TRUE(proto.tensor_content().empty());
    Tensor tensor = Tensor();
    ASSERT_TRUE(tensor.FromProto(proto));
    EXPECT_EQ(tensor.dtype(), DT_HALF);
    EXPECT_EQ(tensor.shape(), TensorShape({2, 2}));
    auto* data = tensor.flat<Eigen::half>().data();
    ASSERT_EQ(tensor.NumElements(), 4);
    for (auto i = 0; i < tensor.NumElements(); ++i) {
      EXPECT_EQ(data[i], Eigen::half(i));
    }
  }
  ASSERT_EQ(const_cnt, 1);
}

TEST(TaoBaCEReformatPassTest, TestSkipByType) {
  auto g = CreateGraph("DT_INT32");
  RunPass(&g, 1, 1, 1);
  int const_cnt = 0;
  for (auto* node : g->op_nodes()) {
    const_cnt += 1;
    EXPECT_EQ(node->type_string(), "Const");
    EXPECT_EQ(node->attrs().Find("dtype")->type(), DT_INT32);
    auto proto = node->attrs().Find("value")->tensor();
    EXPECT_EQ(proto.dtype(), DT_INT32);
  }
  ASSERT_EQ(const_cnt, 1);
}

TEST(TaoBaCEReformatPassTest, TestSkipBySize) {
  auto g = CreateGraph("DT_FLOAT");
  RunPass(&g, 3, 3, 5);
  int const_cnt = 0;
  for (auto* node : g->op_nodes()) {
    const_cnt += 1;
    EXPECT_EQ(node->type_string(), "Const");
    EXPECT_EQ(node->attrs().Find("dtype")->type(), DT_FLOAT);
    auto proto = node->attrs().Find("value")->tensor();
    EXPECT_EQ(proto.dtype(), DT_FLOAT);
  }
  ASSERT_EQ(const_cnt, 1);
}

}  // namespace tao
}  // namespace tensorflow
