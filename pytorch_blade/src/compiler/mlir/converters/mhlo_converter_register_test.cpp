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

#include <gtest/gtest.h>

#include "compiler/mlir/converters/mhlo_conversion.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/script.h>

using namespace torch::blade;
namespace {
// the custom dummy function in TorchScript
at::Tensor dummy_func(at::Tensor input) {
  return input;
}

// Register the dummy function into TorchScript
TORCH_LIBRARY(aten_test, m) {
  m.def("dummy(Tensor self) -> Tensor", dummy_func);
}

bool ConvertDummyOp(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto pt_input = node.input(0);
  auto ml_value = ctx.GetMlirValue(pt_input);
  ctx.value_map[node.output(0)] = ml_value;
  return true;
}

// Register an OpConverter from TorchScript to MLIR
auto mhlo_conversion = MhloConversionPatternRegister().pattern(
    "aten_test::dummy(Tensor self) -> Tensor",
    ConvertDummyOp);
} // namespace

TEST(MhloConverter, Register) {
  const auto text = R"IR(
  graph(%x: Float(1)):
    %y: Float(1) = aten_test::dummy(%x)
    return (%y)
  )IR";
  auto graph = std::make_shared<torch::jit::Graph>();
  parseIR(text, graph.get());
  std::string parsable_mlir;
  std::string pretty_mlir;
  std::string input_dev_str;
  std::string output_dev_str;
  std::tie(parsable_mlir, pretty_mlir, input_dev_str, output_dev_str) =
      ConvertTorchScriptToMhlo(graph);
  const auto want_mlir = R"MLIR(
    # CHECK: func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "x", output_placements = "cpu", outputs = "1"}} {
    # CHECK:   return %arg0 : tensor<?xf32>
  )MLIR";
  CHECK_EQ(input_dev_str, "cpu");
  CHECK_EQ(output_dev_str, "cpu");
  torch::jit::testing::FileCheck().run(want_mlir, pretty_mlir);
}
