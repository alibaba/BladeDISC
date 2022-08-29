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

#include "ltc/disc_compiler/disc_compiler.h"

namespace c10 {
class IValue;
} // namespace c10

namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch

namespace torch_disc {
namespace compiler {

class Timer {
 public:
  Timer(const std::string& msg, int scale)
      : begin_(std::chrono::steady_clock::now()), msg_(msg), scale_(scale) {}
  ~Timer() {
    std::cerr << msg_ << ":"
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - begin_)
                     .count() /
            1000.0 / scale_
              << " ms" << std::endl;
  }

 private:
  std::chrono::steady_clock::time_point begin_;
  std::string msg_;
  int scale_;
};

void DumpProgramAndData(
    const std::shared_ptr<torch::jit::Graph> graph,
    c10::ArrayRef<std::shared_ptr<torch::lazy::BackendData>> arguments,
    const std::string& path);
void LoadAndReplay(const std::string& path, int iters, int warmup = 10);
ExecutablePtr TestPyBind(const std::string& path);
// void Run();

} //  namespace compiler
} //  namespace torch_disc