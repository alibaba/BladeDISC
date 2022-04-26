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

#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <memory>
#include <string>

namespace c10 {
class IValue;
} // namespace c10

namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch

namespace torch_disc {

void SaveTSGraphAsModule(std::shared_ptr<torch::jit::Graph>, std::string);
void SaveTSData(c10::ArrayRef<torch::lazy::BackendDataPtr>, std::string);

} // namespace torch_disc
