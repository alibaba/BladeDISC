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

#include "pytorch_blade/compiler/tensorrt/tensorrt_engine.h"

#include <algorithm>
#include <sstream>

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/backends/engine_interface.h"
#include "pytorch_blade/compiler/tensorrt/tensorrt_engine_context.h"

#include <torch/script.h>

namespace torch {
namespace blade {
namespace tensorrt {

class TRTEngine : public torch::blade::backends::EngineInterface {
 public:
  using State = torch::blade::backends::EngineState;

  DISALLOW_COPY_AND_ASSIGN(TRTEngine);
  TRTEngine(const State& state);

  at::List<at::Tensor> Execute(const at::List<at::Tensor>& inputs) override;

  const State& GetState() const {
    return *engine_state_;
  }

  static const char* GetBackendName() {
    return "TensorRT";
  }
  static std::shared_ptr<TRTEngine> Create(const State& engine_state);
  bool ShouldFallback(const at::List<at::Tensor>& inputs) override;

 private:
  std::shared_ptr<TRTContext> engine_ctx_;
  std::shared_ptr<State> engine_state_;
};

TRTEngine::TRTEngine(const State& state) {
  engine_state_ = std::make_shared<State>(std::move(state));
  engine_ctx_ = std::make_shared<TRTContext>(engine_state_);
}

at::List<at::Tensor> TRTEngine::Execute(const at::List<at::Tensor>& inputs) {
  return engine_ctx_->Execute(inputs);
}

bool TRTEngine::ShouldFallback(const at::List<at::Tensor>& inputs) {
  return !engine_ctx_->IsInRange(inputs);
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
