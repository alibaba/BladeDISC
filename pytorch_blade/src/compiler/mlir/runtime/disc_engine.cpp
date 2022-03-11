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

#include "compiler/mlir/runtime/disc_engine.h"

#include <algorithm>
#include <sstream>
#include "common_utils/logging.h"

namespace torch {
namespace blade {
namespace disc {

DiscEngine::DiscEngine(const State& state) {
  engine_state_ = std::make_shared<State>(std::move(state));
  auto engine_ctx = FetchRalContext();
  CHECK_NOTNULL(engine_ctx);
}

torch::List<torch::Tensor> DiscEngine::Execute(
    const torch::List<torch::Tensor>& inputs) {
  auto engine_ctx = FetchRalContext();
  CHECK_NOTNULL(engine_ctx);
  return engine_ctx->Execute(inputs);
}

// FetchRalContext guarantee to return an effective engine_ctx_
std::shared_ptr<RalContext> DiscEngine::FetchRalContext() {
  // Note: we use lock_guard(mutex) since the multi-threads collision with low
  // frequency.
  std::lock_guard<std::mutex> guard(ctx_lock_);
  if (engine_ctx_ != nullptr) {
    return engine_ctx_;
  }

  engine_ctx_ = std::make_shared<RalContext>(engine_state_);
  return engine_ctx_;
}

void DiscEngine::ReleaseRalContext() {
  std::lock_guard<std::mutex> guard(ctx_lock_);
  if (engine_ctx_ == nullptr) {
    return;
  }
  engine_ctx_.reset();
}

std::shared_ptr<DiscEngine> DiscEngine::Create(const State& engine_state) {
  return std::shared_ptr<DiscEngine>(new DiscEngine(engine_state));
}

static auto torch_blade_engine_creator =
    torch::blade::backends::EngineCreatorRegister().RegisterBackend(
        DiscEngine::GetBackendName(),
        &DiscEngine::Create);

} // namespace disc
} // namespace blade
} // namespace torch
