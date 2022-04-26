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

#include "ltc/disc_backend/debug_utils.h"

#include "compiler/jit/tool_funcs.h"

#include <sys/stat.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/script.h>
#include <fstream>

namespace torch_disc {

void SaveTSGraphAsModule(
    std::shared_ptr<torch::jit::Graph> graph,
    std::string save_path) {
  torch::jit::Module module("__torch__.PlaceholderModule");
  module.register_attribute("training", torch::jit::BoolType::get(), true);
  torch::blade::create_method_from_graph(module, "forward", graph);
  module.save(save_path);
}

void SaveTSData(
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    std::string save_path) {
  std::vector<c10::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data =
        std::static_pointer_cast<torch::lazy::TSData>(argument);
    if (ts_data->scalar.has_value()) {
      stack.emplace_back(ts_data->scalar.value());
    } else {
      stack.emplace_back(ts_data->data());
    }
  }

  TORCH_CHECK(
      !mkdir(save_path.c_str(), 0755), "unable to create dir: " + save_path);

  for (size_t k = 0; k < stack.size(); ++k) {
    auto fname = save_path + "/" + std::to_string(k) + ".pt";
    auto chars = torch::jit::pickle_save(stack[k]);
    std::ofstream ofstream(fname, std::ios::out | std::ios::binary);
    ofstream.write(chars.data(), chars.size());
    ofstream.close();
  }
}

} // namespace torch_disc
