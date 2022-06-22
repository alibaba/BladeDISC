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

#pragma once

#include <torch/script.h>

namespace torch {
namespace blade {

std::string subgraph_input_name_mangle(const std::string& inp_name);
std::string subgraph_input_name_demangle(const std::string& inp_name);

torch::jit::Node* MergeNodeIntoGroup(
    torch::jit::Node* group,
    torch::jit::Node* n);
torch::TypePtr get_list_tensor_type();
torch::TypePtr create_tensor_type_from_scalar_type(const c10::Type& typ);
void set_tensor_shape(torch::jit::Value* val, const std::vector<int64_t>& dims);
} // namespace blade
} // namespace torch
