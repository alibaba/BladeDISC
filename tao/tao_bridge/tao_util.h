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

#ifndef TAO_TAO_BRIDGE_TAO_UTIL_H_
#define TAO_TAO_BRIDGE_TAO_UTIL_H_

#include "absl/strings/string_view.h"

#include <memory>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {

class Graph;

namespace tao {
namespace util {

bool HasOpType(const Graph& g, absl::string_view op_type);

std::unique_ptr<FunctionLibraryDefinition> ReachableDefinitions(
    const FunctionLibraryDefinition& flib, const FunctionDef& func);

}  // namespace util
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_TAO_UTIL_H_
