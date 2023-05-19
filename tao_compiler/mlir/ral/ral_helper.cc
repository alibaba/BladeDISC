//===- ral_helper.cc ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/ral/ral_helper.h"

#include <iostream>
#include <mutex>
#include <unordered_map>

#include "mlir/ral/ral_logging.h"

namespace tao {
namespace ral {

struct TaoRalApiRegistry::Impl {
  std::mutex mu;
  std::unordered_map<std::string, TaoRalApiRegistry::api_func_t> api_funcs;
};

TaoRalApiRegistry::TaoRalApiRegistry() : impl_(new TaoRalApiRegistry::Impl) {}

TaoRalApiRegistry::~TaoRalApiRegistry() {}

TaoRalApiRegistry& TaoRalApiRegistry::Global() {
  static TaoRalApiRegistry registry;
  return registry;
}

bool TaoRalApiRegistry::Register(const std::string& name,
                                 const std::string& nickname,
                                 api_func_t api_func) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it = impl_->api_funcs.emplace(name, api_func);
  TAO_VLOG(1) << "register ral function: " << name;
  impl_->api_funcs.emplace(nickname, api_func);
  return it.second;
}

TaoRalApiRegistry::api_func_t TaoRalApiRegistry::Find(const std::string& name) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  TaoRalApiRegistry::api_func_t api_func;
  auto it = impl_->api_funcs.find(name);
  if (it != impl_->api_funcs.end()) {
    api_func = it->second;
  }
  return api_func;
}

}  // namespace ral
}  // namespace tao