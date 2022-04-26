// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace torch_disc {
namespace compiler {

class MyMethod : public torch::CustomClassHolder {
 public:
  MyMethod(){};
  torch::Tensor Run(torch::Tensor& a, torch::Tensor& b) {
    auto ret = a + b;
    std::cout << ret << std::endl;
    return ret;
  }
};

static auto my_method_class =
    torch::class_<MyMethod>("torch_disc", "MyMethod")
        .def(
            "Run",
            [](const c10::intrusive_ptr<MyMethod>& self,
               torch::Tensor& a,
               torch::Tensor& b) { return self->Run(a, b); });

// this unit test demonstrates how DiscClass works
// 1. implement a CustomClassHolder
// 2. construct the graph with prim::CallMethod to call this custom class
// 3. prepare input and output Tensors
// 4. execute this Graph with Torch JIT GraphExecutor
TEST(DiscEngineTest, Registry) {
  std::shared_ptr<torch::jit::Graph> g = std::make_shared<torch::jit::Graph>();
  c10::IValue obj = torch::make_custom_class<MyMethod>();
  torch::Tensor ta = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
  torch::Tensor tb = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
  {
    // construct a demo Graph with MyMethod CustomClassHolder:
    //
    //  graph(%input_obj : __torch__.torch.classes.torch_disc.MyMethod,
    //    %p0 : Tensor,
    //    %p1 : Tensor):
    //  %3 : Tensor = prim::CallMethod[name="Run"](%input_obj, %p0, %p1)
    //  return (%3)
    auto input0 = g->addInput("input_obj")->setType(obj.type());
    auto input1 = g->addInput("p0");
    auto input2 = g->addInput("p1");
    auto result =
        g->insertNode(
             g->create(torch::jit::prim::CallMethod, {input0, input1, input2}))
            ->s_(torch::jit::attr::name, std::move("Run"))
            ->output();
    g->registerOutput(result);
  }

  std::vector<torch::jit::IValue> stack;
  torch::jit::GraphExecutor executor(g, "");
  stack.push_back(obj);
  stack.push_back(ta);
  stack.push_back(tb);
  executor.run(stack);
  EXPECT_EQ(stack.size(), 1UL);
  auto ret = ta + tb;
  EXPECT_TRUE(stack[0].toTensor().equal(ret));
}

} // namespace compiler
} // namespace torch_disc
