#ifndef __FUSION_H__
#define __FUSION_H__

#include <torch/script.h>

namespace torch {
namespace blade {

std::string subgraph_input_name_mangle(const std::string& inp_name);
std::string subgraph_input_name_demangle(const std::string& inp_name);

torch::jit::Node* MergeNodeIntoGroup(
    torch::jit::Node* group,
    torch::jit::Node* n);
torch::TypePtr get_list_tensor_type();
} // namespace blade
} // namespace torch
#endif // __FUSION_H__
