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

#include "decoupling/tao_compiler_trace.h"

#include <fstream>
#include <mutex>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

TaoCompilerTrace::TaoCompilerTrace() {
  ReadStringFromEnvVar("TAO_COMPILER_TRACE_DUMP_PATH", "", &dump_path_);
  activated_.store(!dump_path_.empty());
}

void TaoCompilerTrace::Shutdown() {
  if (!activated_.load()) {
    return;
  }
  std::ofstream file(dump_path_);
  constexpr char sep = '\001';
  std::lock_guard<std::mutex> guard(events_lock_);
  for (auto& event : events_) {
    file << event.timestamp_us << sep << event.key << sep << event.value
         << std::endl;
  }
}

void TaoCompilerTrace::OnEvent(std::string&& key, std::string&& value) {
  if (!activated_.load()) {
    return;
  }
  auto timestamp = Env::Default()->NowMicros();
  std::lock_guard<std::mutex> guard(events_lock_);
  events_.emplace_back(Event{timestamp, std::move(key), std::move(value)});
}

/* static */ TaoCompilerTrace* TaoCompilerTrace::Instance() {
  static TaoCompilerTrace instance;
  return &instance;
}

}  // namespace tao
}  // namespace tensorflow
