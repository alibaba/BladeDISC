#include "onnx_funcs.h"
#include "common_utils/logging.h"

namespace torch {
namespace addons {
using namespace torch::jit;

// For tensorrt7.0, constant operator supports only data types: INT32, FLOAT
void CastDownAllConstantDoubleToFloat(Block* block) {
  auto it = block->nodes().begin();
  for (; it != block->nodes().end(); ++it) {
    auto node = *it;
    for (auto block : node->blocks()) {
      CastDownAllConstantDoubleToFloat(block);
    }

    if (node->kind() == ::c10::onnx::Constant) {
      auto val = node->t(attr::value);
      torch::ScalarType dtype = val.scalar_type();
      if (dtype == torch::ScalarType::Double) {
        val = val.to(torch::ScalarType::Float);
        node->removeAttribute(attr::value);
        node->t_(attr::value, val);
      }
    }
  }
}

void CastDownAllConstantDoubleToFloat(std::shared_ptr<Graph> graph) {
  CastDownAllConstantDoubleToFloat(graph->block());
}
} // namespace addons
} // namespace torch
