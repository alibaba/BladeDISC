#include <memory>
#include <tuple>

namespace torch {
namespace jit {
class Graph;
class Node;
} // namespace jit
} // namespace torch

namespace torch {
namespace addons {

bool IsMlirMhloSupported(const torch::jit::Node&);

// Return a pair of the MLIR module strings, with the first one in
// parsable/compilable format and the second one in pretty format which elide
// large elements like constant tensors.
std::tuple<std::string, std::string, std::string, std::string>
ConvertTorchScriptToMhlo(std::shared_ptr<torch::jit::Graph> graph);
} // namespace addons
} // namespace torch
