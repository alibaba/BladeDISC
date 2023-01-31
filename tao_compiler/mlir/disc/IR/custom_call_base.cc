/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file defines the basic macros for custom call.

#include "mlir/disc/IR/custom_call_base.h"

#include <unordered_map>

namespace mlir {
namespace mhlo_disc {

struct CustomCallRegistry::Impl {
  std::mutex mu;
  std::unordered_map<std::string, CustomCallRegistry::reify_shapes_func_t>
      reify_shapes_funcs;
  std::unordered_map<std::string,
                     CustomCallRegistry::lower_to_library_call_func_t>
      lower_to_library_call_funcs;
};

CustomCallRegistry::CustomCallRegistry()
    : impl_(new CustomCallRegistry::Impl) {}

CustomCallRegistry::~CustomCallRegistry() {}

CustomCallRegistry& CustomCallRegistry::Global() {
  static CustomCallRegistry registry;
  return registry;
}

bool CustomCallRegistry::Register(
    const std::string& name, reify_shapes_func_t reify_shapes_func,
    lower_to_library_call_func_t lower_to_library_call_func) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it_reify_shapes_funcs =
      impl_->reify_shapes_funcs.emplace(name, reify_shapes_func);
  auto it_lower_to_library_call_funcs =
      impl_->lower_to_library_call_funcs.emplace(name,
                                                 lower_to_library_call_func);
  return it_reify_shapes_funcs.second && it_lower_to_library_call_funcs.second;
}

CustomCallRegistry::reify_shapes_func_t CustomCallRegistry::FindReifyShapesFunc(
    const std::string& name) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  CustomCallRegistry::reify_shapes_func_t func;
  auto it = impl_->reify_shapes_funcs.find(name);
  if (it != impl_->reify_shapes_funcs.end()) {
    func = it->second;
  }
  return func;
}

CustomCallRegistry::lower_to_library_call_func_t
CustomCallRegistry::FindLowerToLibraryCallFunc(const std::string& name) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  CustomCallRegistry::lower_to_library_call_func_t func;
  auto it = impl_->lower_to_library_call_funcs.find(name);
  if (it != impl_->lower_to_library_call_funcs.end()) {
    func = it->second;
  }
  return func;
}

}  // namespace mhlo_disc
}  // namespace mlir
