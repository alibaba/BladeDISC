#ifndef __JIT_ONNX_H__
#define __JIT_ONNX_H__
#include <torch/script.h>
namespace torch {
namespace blade {
void CastDownAllConstantDoubleToFloat(std::shared_ptr<torch::jit::Graph> graph);
}
} // namespace torch
#endif //__JIT_ONNX_H__
