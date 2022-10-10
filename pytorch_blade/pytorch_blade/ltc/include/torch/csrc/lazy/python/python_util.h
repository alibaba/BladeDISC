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
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

#include <vector>

namespace torch {
namespace lazy {

c10::optional<SourceLocation> TORCH_API GetPythonFrameTop();

std::vector<SourceLocation> TORCH_API GetPythonFrames();

} // namespace lazy
} // namespace torch
