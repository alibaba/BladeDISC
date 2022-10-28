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
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/script.h>
#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"

namespace torch {
namespace blade {

bool cudaAvailable() {
#if TORCH_BLADE_BUILD_WITH_CUDA
  return true;
#else
  return false;
#endif
}

void eraseInputShape(const std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto input : graph->inputs()) {
    if (auto type = input->type()->cast<c10::TensorType>()) {
      input->setType(type->dimensionedOnly());
    }
  }
}

void autoInputDevice(const std::shared_ptr<torch::jit::Graph>& graph) {
#if PYTORCH_VERSION_GE(1, 12)
  for (auto input : graph->inputs()) {
    if (auto type = input->type()->cast<c10::TensorType>()) {
      if (type->device())
        continue;
      if (cudaAvailable()) {
        input->setType(type->withDevice(c10::Device(c10::kCUDA, 0)));
      } else {
        input->setType(type->withDevice(c10::Device(c10::kCPU)));
      }
    }
  }
#endif
}

// FILE_CHECK parses the input graph string and fill the current
// device to inputs if no default device, check the static
// shape pattern(s_pattern) and dynamic shape pattern (dy_patter)
// with Pytorch FileCheck API
#define FILE_CHECK(graph_str, s_pattern, dy_pattern)          \
  auto g = std::make_shared<torch::jit::Graph>();             \
  torch::jit::parseIR(graph_str, g.get());                    \
  autoInputDevice(g);                                         \
  torch::blade::PropagateInputShapes(g);                      \
  torch::jit::testing::FileCheck().check(s_pattern)->run(*g); \
  eraseInputShape(g);                                         \
  torch::blade::PropagateInputShapes(g);                      \
  torch::jit::testing::FileCheck().check(dy_pattern)->run(*g);

#if PYTORCH_VERSION_GE(1, 8)
TEST(PropagateInputShapesTest, SimpleUnary) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(3, 4, 5, strides=[20, 5, 1], device=cuda:0)):
  %1 : Tensor = aten::relu(%p1)
  return (%1)
)IR";
  const std::string s_pattern =
      "Float(3, 4, 5, strides=[20, 5, 1], device=cuda:0) = aten::relu(%p1)";
  const std::string d_pattern =
      "Float(*, *, *, device=cuda:0) = aten::relu(%p1)";
  FILE_CHECK(graph_str, s_pattern, d_pattern);
}
#endif

#if PYTORCH_VERSION_GE(1, 12)
TEST(PropagateInputShapesTest, SliceOp) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(1, 512, strides=[512, 1], device=cpu)):
  %1 : int = prim::Constant[value=0]()
  %2 : int = prim::Constant[value=1]()
  %3 : Tensor = aten::slice(%p1, %1, %1, %2, %2)
  return (%3)
)IR";
  const std::string s_pattern =
      "%3 : Float(1, 512, strides=[512, 1], device=cpu) = aten::slice(%p1, %1, %1, %2, %2)";
  const std::string d_pattern =
      "%3 : Float(*, *, device=cpu) = aten::slice(%p1, %1, %1, %2, %2)";
  FILE_CHECK(graph_str, s_pattern, d_pattern);
}

TEST(PropagateInputShapesTest, AddOp) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(48, 128, 768, strides=[98304, 768, 1], requires_grad=0),
      %p2 : Float(1, 128, 768, strides=[98304, 768, 1], requires_grad=0),
      %p3 : int):
  %1 : Tensor = aten::add(%p1, %p2, %p3)
  return (%1)
)IR";

#if TORCH_BLADE_BUILD_WITH_CUDA
  const std::string s_pattern =
      "%3 : Float(48, 128, 768, strides=[98304, 768, 1], device=cuda:0) = aten::add(%p1, %p2, %p3)";
  const std::string d_pattern =
      "%3 : Float(*, *, *, device=cuda:0) = aten::add(%p1, %p2, %p3)";
#else
  const std::string s_pattern =
      "%3 : Float(48, 128, 768, strides=[98304, 768, 1], device=cpu) = aten::add(%p1, %p2, %p3)";
  const std::string d_pattern =
      "%3 : Float(*, *, *, device=cpu) = aten::add(%p1, %p2, %p3)";
#endif
  FILE_CHECK(graph_str, s_pattern, d_pattern);
}

TEST(PropagateInputShapesTest, AtenEmbeddingDenseBackward) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(8, 512, 768, strides=[393216, 768, 1], device=cuda:0),
      %p2 : Long(8, 512, strides=[512, 1], requires_grad=0, device=cuda:0)):
  %2 : int = prim::Constant[value=28996]()
  %3 : int = prim::Constant[value=0]()
  %4 : bool = prim::Constant[value=0]()
  %5 : Tensor = aten::embedding_dense_backward(%p1, %p2, %2, %3, %4)
  return (%5)
)IR";
  const std::string s_pattern =
      "%5 : Float(28996, 768, strides=[768, 1], device=cuda:0) = aten::embedding_dense_backward(%p1, %p2, %2, %3, %4)";
  const std::string d_pattern =
      "%5 : Float(28996, *, device=cuda:0) = aten::embedding_dense_backward(%p1, %p2, %2, %3, %4)";
  FILE_CHECK(graph_str, s_pattern, d_pattern);
}

TEST(PropagateInputShapesTest, AtenTanhBackward) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(8, 512, 768, strides=[393216, 768, 1], device=cuda:0),
      %p2 : Float(8, 512, 768, strides=[393216, 768, 1], device=cuda:0)):
  %1 : Tensor = aten::tanh_backward(%p1, %p2)
  return (%1)
)IR";
  const std::string s_pattern =
      "%2 : Float(8, 512, 768, strides=[393216, 768, 1], device=cuda:0) = aten::tanh_backward(%p1, %p2)";
  const std::string d_pattern =
      "%2 : Float(*, *, *, device=cuda:0) = aten::tanh_backward(%p1, %p2)";
  FILE_CHECK(graph_str, s_pattern, d_pattern);
}

#endif

} //  namespace blade
} //  namespace torch
