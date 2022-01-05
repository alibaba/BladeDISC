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

#include "tao_bridge/tao_util.h"

#include <string>
#include <utility>
#include <vector>

#include "tao_bridge/test_helpers.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tao {
namespace util {

using FDH = FunctionDefHelper;

namespace {

void CreateGraph(Graph* g, const std::string& gdef_ascii) {
  GraphDef gdef;
  CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef));
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef, g));
}

REGISTER_OP("ConstMock").Output("output: float").Attr("value: tensor")
    /* .Attr("dtype: type") */;

REGISTER_OP("MatMulMock")
    .Input("a: float")
    .Input("b: float")
    .Output("product: float")
    /* .Attr("T: {float}") */;

REGISTER_OP("NoOp");

GraphDef GDef(std::vector<NodeDef> nodes, std::vector<FunctionDef> funcs) {
  GraphDef g;
  VersionDef* versions = g.mutable_versions();
  versions->set_producer(TF_GRAPH_DEF_VERSION);
  versions->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
  for (const auto& n : nodes) {
    *(g.add_node()) = n;
  }
  auto lib = g.mutable_library();
  for (const auto& f : funcs) {
    *(lib->add_function()) = f;
  }
  return g;
}

// Helper to construct a NodeDef.
NodeDef NDef(std::string name, std::string op, std::vector<std::string> inputs,
             std::vector<std::pair<std::string, FDH::AttrValueWrapper>> attrs,
             const std::string& device) {
  NodeDef n;
  n.set_name(string(name));
  n.set_op(string(op));
  for (const auto& in : inputs) n.add_input(in);
  n.set_device(device);
  for (auto na : attrs) n.mutable_attr()->insert({na.first, na.second.proto});
  return n;
}

}  // namespace

TEST(TaoUtils, HasOpType) {
  Graph g(OpRegistry::Global());
  CreateGraph(&g,
              "node { name: 'A' op: 'ConstMock' }"
              "node { name: 'B' op: 'ConstMock' }"
              "node { name: 'C' op: 'MatMulMock' input: [ 'A:0', 'B:0' ] }");

  EXPECT_TRUE(HasOpType(g, "ConstMock"));
  EXPECT_TRUE(HasOpType(g, "MatMulMock"));
  EXPECT_FALSE(HasOpType(g, "Sum"));
}

TEST(TaoUtils, ReachableDefinitions) {
  const auto make_simple_fdef = [](const string& name) {
    auto func_def = FDH::Create(
        name, {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
        {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
        /* Mapping between function returns and function node outputs. */
        {{"z", "output:z:0"}});
    return func_def;
  };

  const auto make_complex_fdef = [](const string& name,
                                    const string& call_name) {
    auto func_def = FDH::Create(
        name, {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
        {{{"output"}, call_name, {"x", "y"}, {{"T", "$T"}}}},
        /* Mapping between function returns and function node outputs. */
        {{"z", "output:z:0"}});
    return func_def;
  };

  FunctionDef func_1 = make_simple_fdef("Func1");
  FunctionDef func_2 = make_simple_fdef("Func2");
  FunctionDef func_3 = make_complex_fdef("Func3", "Func1");

  constexpr char kDevice[] = "/device:CPU:0";
  GraphDef graph = GDef(
      {
          NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
          NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
          NDef("x", "Func1", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
          NDef("y", "Func3", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
      },
      // FunctionLib
      {func_1, func_2, func_3});

  FunctionLibraryDefinition flib(OpRegistry::Global(), graph.library());

  auto pfunc_1 = flib.Find("Func1");
  EXPECT_TRUE(pfunc_1 != nullptr);
  auto pfunc_2 = flib.Find("Func2");
  EXPECT_TRUE(pfunc_2 != nullptr);
  auto pfunc_3 = flib.Find("Func3");
  EXPECT_TRUE(pfunc_3 != nullptr);
  auto pfunc_not_exist = flib.Find("not_exist");
  EXPECT_TRUE(pfunc_not_exist == nullptr);

  // test simple function
  {
    auto new_flib = ReachableDefinitions(flib, *pfunc_1);
    EXPECT_TRUE(new_flib->Contains("Func1"));
    EXPECT_FALSE(new_flib->Contains("Func2"));
    EXPECT_FALSE(new_flib->Contains("Func3"));
  }

  // test complex function
  {
    auto new_flib = ReachableDefinitions(flib, *pfunc_3);
    EXPECT_TRUE(new_flib->Contains("Func1"));
    EXPECT_FALSE(new_flib->Contains("Func2"));
    EXPECT_TRUE(new_flib->Contains("Func3"));
  }
}

}  // namespace util
}  // namespace tao
}  // namespace tensorflow
