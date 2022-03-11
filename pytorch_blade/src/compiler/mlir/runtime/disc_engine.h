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
#include <torch/script.h>
#include "compiler/backends/engine_interface.h"
#include "compiler/mlir/runtime/ral_context.h"

namespace torch {
namespace blade {
namespace disc {
class DiscEngine : public torch::blade::backends::EngineInterface {
 public:
  using State = torch::blade::backends::EngineState;

  DISALLOW_COPY_AND_ASSIGN(DiscEngine);
  DiscEngine(const State& state);

  torch::List<torch::Tensor> Execute(
      const torch::List<torch::Tensor>& inputs) override;

  const State& GetState() const {
    return *engine_state_;
  }

  static const char* GetBackendName() {
    return "DISC";
  }
  static std::shared_ptr<DiscEngine> Create(const State& engine_state);

 private:
  std::shared_ptr<RalContext> FetchRalContext();
  void ReleaseRalContext();

  // use these fields carefully in multi-threads context
  std::mutex ctx_lock_;
  // don't use it directly, please use FetchRalContext
  std::shared_ptr<RalContext> engine_ctx_;
  std::shared_ptr<State> engine_state_;
};
} // namespace disc
} // namespace blade
} // namespace torch
