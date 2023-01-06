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

#include <dlfcn.h>

#include <functional>
#include <iostream>

#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/pdll_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_base.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"

#ifdef TAO_RAL_USE_STREAM_EXECUTOR

namespace tao {
namespace ral {
namespace gpu {

namespace se = ::stream_executor;

namespace se_impl {

using namespace tensorflow;

namespace gpu_conv_impl {}  // namespace gpu_conv_impl
}  // namespace se_impl

}  // namespace gpu
}  // namespace ral
}  // namespace tao
#endif
