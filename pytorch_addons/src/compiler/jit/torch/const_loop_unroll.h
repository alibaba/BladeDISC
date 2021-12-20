#pragma once
#include <torch/csrc/jit/ir/ir.h>
namespace torch {
namespace addons {
void UnrollConstLoops(std::shared_ptr<torch::jit::Graph>& graph);
}
} // namespace torch
