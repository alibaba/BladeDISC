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

#ifndef TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_
#define TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/tsl/platform/default/integral_types.h"

namespace tensorflow {
namespace tao {

// Trace events in tao compiler.
class TaoCompilerTrace {
 private:
  TaoCompilerTrace();

 public:
  void OnEvent(std::string&& key, std::string&& value);
  void Shutdown();

  static TaoCompilerTrace* Instance();

 private:
  struct Event {
    uint64_t timestamp_us;
    std::string key;
    std::string value;
  };

  std::mutex events_lock_;
  std::vector<Event> events_ TF_GUARDED_BY(
      events_lock_);  // Bookkeeping all the events in memory (for now).
  std::atomic<bool> activated_{false};  // Whether tracing is activated.
  std::string dump_path_ = "";          // File path to dump the events.

  TF_DISALLOW_COPY_AND_ASSIGN(TaoCompilerTrace);
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_
