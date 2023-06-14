//===- ral_context.cc ----------------------===//
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

#include "mlir/ral/ral_context.h"

#include <array>
#include <atomic>
#include <iostream>
#include <thread>

#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"
#include "mlir/ral/ral_logging.h"

namespace tao {
namespace ral {

namespace {

std::atomic<int> nextThreadIdx{0};

}

const char* kRalRecvInput = "ral_recv_input";
const char* kRalSendOutput = "ral_send_output";
const char* kRalCudaConst = "ral_const";
const char* kRalHostConst = "ral_const";
const char* kRalBitcast = "inc_ref";

int ThreadLocalIndex::Get() {
  static thread_local int index = nextThreadIdx++;
  return index;
}

struct Context::Impl {
  struct Status {
    status_t errcode = 0;
    std::string err_msg;

    Status& merge(const Status& st) {
      // Only update status when current status is ok.
      if (this->ok()) {
        this->errcode = st.errcode;
        this->err_msg = st.err_msg;
      }
      return *this;
    }

    void clear() {
      errcode = 0;
      err_msg.clear();
    }

    bool ok() const { return errcode == 0; }
  };
  // status cross multi-threads
  Status global_status;
  // per-thread message storage
  thread_local static std::string local_err_msg;
  std::unordered_map<std::string, std::unique_ptr<Driver>> drivers;
  std::unordered_map<std::string, std::shared_ptr<Resource>> resources;

  // using a cache for fast api func lookup when possible.
  using ApiFuncCache =
      std::unordered_map<const char*, TaoRalApiRegistry::api_func_t>;
  std::mutex api_func_cache_mu;
  std::unordered_map<std::thread::id, ApiFuncCache> api_func_cache_map;

  static constexpr const int kMaxNumThreadsAllowed = 1024;
  std::array<ApiFuncCache, kMaxNumThreadsAllowed> fast_api_func_cache_map;

  ApiFuncCache* GetCache() {
    auto tid = ThreadLocalIndex::Get();
    if (tid < kMaxNumThreadsAllowed) {
      return &fast_api_func_cache_map[tid];
    }
    std::lock_guard<std::mutex> l(api_func_cache_mu);
    return &api_func_cache_map[std::this_thread::get_id()];
  }
};

thread_local std::string Context::Impl::local_err_msg;

Context::Context() : impl_(new Impl) {}

Context::~Context() {
  for (auto& resource : impl_->resources) {
    resource.second->onContextFinish(this);
  }
}

void Context::call(const std::string& api_name, void** args) {
  auto api_func = TaoRalApiRegistry::Global().Find(api_name);
  if (!api_func) {
    if (api_name.substr(0, 9) == "ral_debug") {
      TAO_VLOG(0) << "[[DEBUG]] " << api_name;
      return;
    }
    signalError(FAILURE, "api_func " + api_name + " not found");
    return;
  }
  TAO_VLOG(1) << "before call api_func " << api_name;
  api_func(args);
  TAO_VLOG(1) << "after call api_func " << api_name;
}

void Context::call(const char* api_name, void** args) {
  auto api_map = impl_->GetCache();
  auto it = api_map->find(api_name);
  if (it != api_map->end()) {
    TAO_VLOG(1) << "before call cached api_func " << api_name;
    it->second(args);
    TAO_VLOG(1) << "after call cached api_func " << api_name;
    return;
  }

  std::string api_name_str = api_name;
  auto api_func = TaoRalApiRegistry::Global().Find(api_name_str);
  if (!api_func) {
    signalError(FAILURE, "api_func " + api_name_str + " not found");
    return;
  }
  (*api_map)[api_name] = api_func;

  TAO_VLOG(1) << "before call api_func " << api_name;
  api_func(args);
  TAO_VLOG(1) << "after call api_func " << api_name;
}

api_func_t Context::find(const std::string& api_name) {
  return TaoRalApiRegistry::Global().Find(api_name);
}

void Context::onExecutionStart(ExecutionContext* exec_ctx) {
  std::lock_guard<std::mutex> lock(mu);
  for (auto& resource : impl_->resources) {
    resource.second->onExecutionStart(exec_ctx);
  }
}

void Context::onExecutionFinish(ExecutionContext* exec_ctx) {
  std::lock_guard<std::mutex> lock(mu);
  for (auto& resource : impl_->resources) {
    resource.second->onExecutionFinish(exec_ctx);
  }
}

status_t Context::getLastError(const char** msg_ptr) {
  std::lock_guard<std::mutex> lock(mu);
  if (msg_ptr) {
    // copy global error message to thread local message in case
    // gloabl status may be modified by other thread, and thus
    // it's not safe to return global err_str's c_str to client.
    impl_->local_err_msg = impl_->global_status.err_msg;
    *msg_ptr = impl_->local_err_msg.c_str();
  }
  return impl_->global_status.errcode;
}

void Context::signalError(status_t errcode, const std::string& err_msg) {
  if (errcode == 0) return;
  std::lock_guard<std::mutex> lock(mu);
  signalErrorLocked(errcode, err_msg);
}

void Context::clearError() {
  std::lock_guard<std::mutex> lock(mu);
  impl_->global_status.clear();
}

void Context::signalErrorLocked(status_t errcode, const std::string& err_msg) {
  Impl::Status st;
  st.errcode = errcode;
  st.err_msg = err_msg;
  TAO_LOG(FATAL) << "[[ERROR]] Context catch an exception: " << err_msg;
  impl_->global_status.merge(st);
}

void Context::addDriver(const std::string& name,
                        std::unique_ptr<Driver> driver) {
  std::lock_guard<std::mutex> lock(mu);
  impl_->drivers.emplace(name, std::move(driver));
}

Driver* Context::getDriver(const std::string& name) {
  std::lock_guard<std::mutex> lock(mu);
  auto it = impl_->drivers.find(name);
  if (it != impl_->drivers.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

std::shared_ptr<Context::Resource> Context::getOrCreateResource(
    const std::string& key, std::function<Resource*()> creator) {
  std::lock_guard<std::mutex> lock(mu);
  if (!impl_->global_status.ok()) {
    return nullptr;
  }
  auto it = impl_->resources.find(key);
  if (it == impl_->resources.end()) {
    Resource* r = creator();
    if (!r) {
      signalErrorLocked(FAILURE, "resource creation failed: " + key);
      return nullptr;
    }
    it = impl_->resources
             .insert(std::make_pair(key, std::shared_ptr<Resource>(r)))
             .first;
  }
  return it->second;
}

struct ExecutionContext::Impl {
  Context* context;
};

ExecutionContext::ExecutionContext(Context* context)
    : impl_(new Impl{context}) {}

ExecutionContext::~ExecutionContext() {}

Context* ExecutionContext::getContext() { return impl_->context; }

}  // namespace ral
}  // namespace tao
