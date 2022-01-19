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

/*
This file is derived from PyTorch, and the following modifications are made to
meet our requirements:

1. return type of `ToONNX` changed from std::shared_ptr<Graph> to
   std::tuple<std::shared_ptr<Graph>, std::unordered_map<Value*, Value*>>
   since we need to obtain Value mapping relationships between torchscript graph
   and onnx graph.
2. argument `env` of `BlockToONNX` is now passed by reference instead of passed
by value.
*/
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace torch {
namespace blade {

std::tuple<
    std::shared_ptr<torch::jit::Graph>,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>>
ToONNX(
    std::shared_ptr<torch::jit::Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type);
void BlockToONNX(
    torch::jit::Block* old_block,
    torch::jit::Block* new_block,
    torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& env);
} // namespace blade
} // namespace torch
