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

#ifndef __JIT_PYBIND_ONNX_H__
#define __JIT_PYBIND_ONNX_H__

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace torch {
namespace addons {

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
} // namespace addons
} // namespace torch
#endif //__JIT_PYBIND_ONNX_H__