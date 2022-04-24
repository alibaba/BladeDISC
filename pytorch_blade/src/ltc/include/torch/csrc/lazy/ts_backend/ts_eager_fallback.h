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

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

#include <functional>

namespace torch {
namespace lazy {

bool force_eager_fallback(c10::Symbol op);
void ltc_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

void ts_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    c10::DeviceType device_type);

// The TorchScript backend does not register itself with pytorch dispatcher
// until it is explicitly initialized.  This function should only be called
// by the main Torchscript backend init function.
void register_ts_ltc_eager_fallback();

} // namespace lazy
} // namespace torch
