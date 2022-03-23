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
  void Run() { std::cerr << "MyMethod::Run" << std::endl; }
};

static auto my_method_class =
    torch::class_<MyMethod>("torch_disc", "MyMethod")
        .def("Run", [](const c10::intrusive_ptr<MyMethod>& self) {
          return self->Run();
        });

TEST(DiscEngineTest, Registry) {
  std::shared_ptr<torch::jit::Graph> g;
  g.reset(new torch::jit::Graph());
  c10::IValue obj = torch::make_custom_class<MyMethod>();
  auto input = g->addInput("input_obj")->setType(obj.type());
  auto result = g->insertNode(g->create(torch::jit::prim::CallMethod, {input}))
                    ->s_(torch::jit::attr::name, std::move("Run"))
                    ->output();
  EXPECT_TRUE(false) << g->toString() << std::endl;
  std::vector<torch::jit::IValue> stack;
  stack.push_back(obj);

  torch::jit::GraphExecutor executor(g, "");
  executor.run(stack);
}

}  // namespace compiler
}  // namespace torch_disc