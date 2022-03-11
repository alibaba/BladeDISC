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

#include "compiler/tensorrt/tensorrt_engine.h"

#include <algorithm>
#include <sstream>

#include "common_utils/logging.h"
#include "compiler/backends/engine_interface.h"
#include "compiler/tensorrt/tensorrt_engine_context.h"

#include <torch/script.h>

namespace torch {
namespace blade {
namespace tensorrt {

class TRTEngine : public torch::blade::backends::EngineInterface {
 public:
  using State = torch::blade::backends::EngineState;

  DISALLOW_COPY_AND_ASSIGN(TRTEngine);
  TRTEngine(const State& state);

  torch::List<torch::Tensor> Execute(
      const torch::List<torch::Tensor>& inputs) override;

  const State& GetState() const {
    return *engine_state_;
  }

  static const char* GetBackendName() {
    return "TensorRT";
  }
  static std::shared_ptr<TRTEngine> Create(const State& engine_state);

 private:
  std::shared_ptr<TRTContext> FetchTRTContext();
  void ReleaseTRTContext();

  // use these fields carefully in multi-threads context
  std::mutex trt_ctx_lock_;
  // don't use it directly, please use FetchTRTContext
  std::shared_ptr<TRTContext> engine_ctx_;
  std::shared_ptr<State> engine_state_;
};

TRTEngine::TRTEngine(const State& state) {
  engine_state_ = std::make_shared<State>(std::move(state));
  auto engine_ctx = FetchTRTContext();
  CHECK_NOTNULL(engine_ctx);
}

torch::List<torch::Tensor> TRTEngine::Execute(
    const torch::List<torch::Tensor>& inputs) {
  auto engine_ctx = FetchTRTContext();
  CHECK_NOTNULL(engine_ctx);
  return engine_ctx->Execute(inputs);
}

// FetchTRTContext guarantee to return an effective engine_ctx_
std::shared_ptr<TRTContext> TRTEngine::FetchTRTContext() {
  // Note: we use lock_guard(mutex) since the multi-threads collision with low
  // frequency.
  std::lock_guard<std::mutex> guard(trt_ctx_lock_);
  if (engine_ctx_ != nullptr) {
    return engine_ctx_;
  }

  engine_ctx_ = std::make_shared<TRTContext>(engine_state_);
  return engine_ctx_;
}

void TRTEngine::ReleaseTRTContext() {
  std::lock_guard<std::mutex> guard(trt_ctx_lock_);
  if (engine_ctx_ == nullptr) {
    return;
  }
  engine_ctx_.reset();
}

std::shared_ptr<TRTEngine> TRTEngine::Create(const State& engine_state) {
  return std::shared_ptr<TRTEngine>(new TRTEngine(engine_state));
}

const char* GetBackendName() {
  return TRTEngine::GetBackendName();
}

static auto torch_blade_engine_creator =
    torch::blade::backends::EngineCreatorRegister().RegisterBackend(
        TRTEngine::GetBackendName(),
        &TRTEngine::Create);

} // namespace tensorrt
} // namespace blade
} // namespace torch
