#ifndef __JIT_TOOL_FUNCS_H__
#define __JIT_TOOL_FUNCS_H__
#include <torch/script.h>

namespace torch {
namespace addons {
void RegisterShapeRecordOp();
void unsafe_remove_method(torch::jit::Module& self, const std::string& name);

torch::jit::Module clone_cpp_module(torch::jit::Module& self);

// A tool to create method from a TorchScript graph,
// and add it to ScriptModule.
void create_method_from_graph(
    torch::jit::Module& self,
    const std::string& name,
    const std::shared_ptr<torch::jit::Graph>& graph);

void register_attr(
    torch::jit::Module& module,
    const std::string& name,
    torch::jit::Module& new_module);

torch::jit::Node* create_get_attr_node(
    torch::jit::Graph* graph,
    torch::jit::Value* obj,
    const std::string& field);

bool is_concrete_shape_tensor_type(const torch::jit::Value& val);
bool is_gpu_tensor_type(const torch::jit::Value& val);

void set_value_type(
    torch::jit::Value& val,
    const std::vector<int64_t>& shape_vec,
    const std::vector<int64_t>& stride_vec,
    const std::string& device_str,
    int64_t scalar_type,
    bool require_grad,
    bool is_contiguous);

std::string node_schema_str(const torch::jit::Node& node);
bool cast_to_tensor_type(torch::jit::Value& value);
} // namespace addons
} // namespace torch
#endif
