//===- ral_context.h ----------------------===//
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

#ifndef RAL_RAL_CONTEXT_H_
#define RAL_RAL_CONTEXT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "mlir/xla/ral/ral_base.h"

namespace tao {
namespace ral {

extern const char* kRalRecvInput;
extern const char* kRalSendOutput;
extern const char* kRalCudaConst;
extern const char* kRalHostConst;
extern const char* kRalBitcast;

// Abstraction of a core device driver api set
class Driver;

struct ThreadLocalIndex {
  // Returns a unique index for each thread.
  static int Get();
};

// Context wrapper for a single execution
class ExecutionContext;

// A warp of runtime environment, containing everything needed to
// execute the program that is compiled by tao mlir compiler.
//
// Resource initialization is implicit (e.g. loading cubin).
// One thing worth noting is that a context may be shared among
// different executions of a compiled program, thus the overhead of
// initialization can be amortized.
class Context {
 public:
  Context();
  virtual ~Context();

  enum { SUCCESS = 0, FAILURE = 1 };

  // Custom call abstraction.
  // Calls a pre-registered custom kernel identified by `name` and
  // forward args to the kernel.
  // TODO: Ideally we should hide all other apis using this api and make the
  // interface clean and stable. This is even more important if we support
  // multi-device in the future (since device is pluggable, one evironment
  // has suport for device type `A` may not also have support for device
  // type `B`).
  virtual void call(const std::string& api_name, void** args);

  void call(const char* api_name, void** args);

  // Returns the api function identified by `api_name` if it exists, otherwise
  // returns nullptr.
  virtual api_func_t find(const std::string& api_name);

  // Returns the status since the last api call.
  // When error occurs, error msg is stored into `err_msg` if it's
  // not null. `err_msg` is empty if status is ok.
  status_t getLastError(const char**);
  // Signals an error to the context, and following api calls will failed.
  void signalError(status_t errcode, const std::string& err_msg);
  // Clears if in error status.
  void clearError();

  // subclass may set up per-execution reosurces using this function
  virtual void onExecutionStart(ExecutionContext*);

  // subclass may clean up per-execution resources using this function
  virtual void onExecutionFinish(ExecutionContext*);

  // Resource abstraction.
  // Registered resource will by managed by the context through it lifetime.
  struct Resource {
    virtual ~Resource() = default;

    virtual void onExecutionStart(ExecutionContext*){};
    virtual void onExecutionFinish(ExecutionContext*){};
    virtual void onContextFinish(Context*){};
  };

  enum LifetimeKind {
    // resoruce is destoried when context is destroies
    kPerContext,

    // resoruce is destoried after an execution of the compiled executable
    kPerExecution,

    // resource is per-process instance
    kGlobal
  };

  // Returns the resource if it has been created, or construts it using the
  // `creator` and then returns it. Returns nullptr if context in error status.
  virtual std::shared_ptr<Resource> getOrCreateResource(
      const std::string& key, std::function<Resource*()> creator);

  // Add a device driver api wrapper for a specific device.
  // If a driver named `name` has already been added, silently ignore this
  // dirver.
  virtual void addDriver(const std::string& name,
                         std::unique_ptr<Driver> driver);
  virtual Driver* getDriver(const std::string& name);

 protected:
  void signalErrorLocked(status_t errcode, const std::string& err_msg);

  struct Impl;
  std::mutex mu;
  std::unique_ptr<Impl> impl_;
};

class OutputBufferWrapper {
 public:
  virtual ~OutputBufferWrapper() {}
  virtual const_buffer_t data() = 0;
  virtual const buffer_shape_t& shape() = 0;
  // Returns true if this wrapper is the exclusive owner
  virtual bool owned() const = 0;
  // mark that this wrapper exclusively owns the underlying buffer.
  virtual void markOwned() = 0;
  // Release the ownership of the wrapper buffer.
  // This requires that the buffer is owned the this wrapper.
  virtual void release() = 0;
};

// Context wrapper for a single execution
class ExecutionContext {
 public:
  ExecutionContext(Context* context);
  virtual ~ExecutionContext();

  Context* getContext();

  // Sends and Receives inputs/outputs from environment.
  // Input/Output buffer may or may not on device. It's the responsibility
  // of client to make sure each input/output buffer meets the compiler
  // reuqirement.
  virtual void bindInput(int input_idx, buffer_t buffer,
                         const buffer_shape_t& shape){};
  virtual void bindOutput(int output_idx,
                          std::unique_ptr<OutputBufferWrapper>* output){};

  template <typename T>
  T* getOrCreateResource(const std::string& key,
                         std::function<Context::Resource*()> creator) {
    auto it = cachedResources_.find(key);
    if (it == cachedResources_.end()) {
      it = cachedResources_
               .emplace(key,
                        getContext()->getOrCreateResource(key, creator).get())
               .first;
    }
    return static_cast<T*>(it->second);
  }

  template <typename T>
  T* getResource(const std::string& key) {
    return getOrCreateResource<T>(key, nullptr);
  }

  void signalError(status_t errcode, const std::string& err_msg) {
    getContext()->signalError(errcode, err_msg);
  }

  template <typename T>
  T* getDriver(const std::string& name) {
    return static_cast<T*>(getContext()->getDriver(name));
  }

  void onExecutionStart() { getContext()->onExecutionStart(this); }

  void onExecutionFinish() { getContext()->onExecutionFinish(this); }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::unordered_map<std::string, Context::Resource*> cachedResources_;
};

template <typename T,
          typename = typename std::enable_if<
              std::is_base_of<ExecutionContext, T>::value>::type,
          typename... Args>
std::unique_ptr<T> MakeExecutionContext(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_CONTEXT_H_
