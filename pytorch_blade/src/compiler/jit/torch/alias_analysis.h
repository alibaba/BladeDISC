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
// TODO: These two methods were added because
// torch::jit::isMutableType is not externally visiable
bool isMutableType(const torch::jit::TypePtr& type);
bool isMutableType(const torch::jit::Value* v);
} // namespace blade
} // namespace torch

