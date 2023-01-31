//===- ral_helper.h ----------------------===//
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

#ifndef RAL_RAL_HELPER_H_
#define RAL_RAL_HELPER_H_

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/xla/ral/ral_base.h"

namespace tao {
namespace ral {

class Context;
class ExecutionContext;

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N>
struct MemRefType {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename T>
struct MemRefType<T, 0> {
  T* basePtr;
  T* data;
  int64_t offset;
};

template <typename T>
struct TaoTypeNameHelper;

#define DEFINE_TAO_TYPE_NAME_HELPER(type, name)  \
  template <>                                    \
  struct TaoTypeNameHelper<type> {               \
    static std::string Invoke() { return name; } \
  };

DEFINE_TAO_TYPE_NAME_HELPER(bool, "i1");
DEFINE_TAO_TYPE_NAME_HELPER(uint8_t, "ui8");
DEFINE_TAO_TYPE_NAME_HELPER(int8_t, "i8");
DEFINE_TAO_TYPE_NAME_HELPER(uint16_t, "ui16");
DEFINE_TAO_TYPE_NAME_HELPER(int16_t, "i16");
DEFINE_TAO_TYPE_NAME_HELPER(uint32_t, "ui32");
DEFINE_TAO_TYPE_NAME_HELPER(int32_t, "i32");
DEFINE_TAO_TYPE_NAME_HELPER(int64_t, "i64");
DEFINE_TAO_TYPE_NAME_HELPER(size_t, "i64");
DEFINE_TAO_TYPE_NAME_HELPER(float, "f32");
DEFINE_TAO_TYPE_NAME_HELPER(double, "f64");
DEFINE_TAO_TYPE_NAME_HELPER(void, "void");
DEFINE_TAO_TYPE_NAME_HELPER(const char*, "pvoid");
DEFINE_TAO_TYPE_NAME_HELPER(char*, "pvoid");
DEFINE_TAO_TYPE_NAME_HELPER(Context*, "pvoid");
DEFINE_TAO_TYPE_NAME_HELPER(ExecutionContext*, "pvoid");
DEFINE_TAO_TYPE_NAME_HELPER(const void*, "pvoid");

template <typename T, int N>
struct TaoTypeNameHelper<MemRefType<T, N>> {
  static inline std::string Invoke() {
    std::ostringstream out;
    out << "m" << N << "d" << TaoTypeNameHelper<T>::Invoke();
    return out.str();
  }
};

template <typename T>
struct TaoTypeNameHelper<T*> {
  static inline std::string Invoke() {
    return "p" + TaoTypeNameHelper<T>::Invoke();
  }
};

template <typename... Remaining>
struct TaoVariadicTypeNameHelper;

template <typename T, typename... Remaining>
struct TaoVariadicTypeNameHelper<T, Remaining...> {
  static inline std::string Invoke() {
    return TaoTypeNameHelper<T>::Invoke() + "_" +
           TaoVariadicTypeNameHelper<Remaining...>::Invoke();
  }
};

template <typename T>
struct TaoVariadicTypeNameHelper<T> {
  static inline std::string Invoke() { return TaoTypeNameHelper<T>::Invoke(); }
};

template <>
struct TaoVariadicTypeNameHelper<> {
  static inline std::string Invoke() { return ""; }
};

template <typename... Ts>
struct TaoTypeNameHelper<std::tuple<Ts...>> {
  static inline std::string Invoke() {
    return TaoVariadicTypeNameHelper<Ts...>::Invoke();
  }
};

template <typename F>
struct TaoRalApiFuncNameHelper;

template <typename Return, typename... Args>
struct TaoRalApiFuncNameHelper<Return (*)(Args...)> {
  static inline std::string Invoke(const std::string& prefix) {
    return prefix + "___" + TaoVariadicTypeNameHelper<Args...>::Invoke() +
           "___" + TaoTypeNameHelper<Return>::Invoke();
  }
};

template <typename Return, typename... Args>
struct TaoRalApiFuncNameHelper<std::function<Return(Args...)>> {
  static inline std::string Invoke(const std::string& prefix) {
    return prefix + "___" + TaoVariadicTypeNameHelper<Args...>::Invoke() +
           "___" + TaoTypeNameHelper<Return>::Invoke();
  }
};

template <typename F, F f>
struct TaoRalApiFuncInvoker;

template <typename R, typename F, typename... ArgTypes>
struct TaoRalApiFuncInvokerImpl;

template <typename R, typename F, typename T, typename... RemainingArgTypes>
struct TaoRalApiFuncInvokerImpl<R, F, T, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    TaoRalApiFuncInvokerImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args + 1, std::forward<ParsedArgs>(parsed_args)..., *(T*)(args[0]));
  }
};

template <typename R, typename F, typename T, int N,
          typename... RemainingArgTypes>
struct TaoRalApiFuncInvokerImpl<R, F, MemRefType<T, N>, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    MemRefType<T, N> memref;
    memref.basePtr = *(T**)(*args++);
    memref.data = *(T**)(*args++);
    memref.offset = *(int64_t*)(*args++);
    for (int i = 0; i < N; ++i) {
      memref.sizes[i] = *(int64_t*)(*args++);
    }
    for (int i = 0; i < N; ++i) {
      memref.strides[i] = *(int64_t*)(*args++);
    }
    TaoRalApiFuncInvokerImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args, std::forward<ParsedArgs>(parsed_args)..., std::move(memref));
  }
};

template <typename R, typename F, typename T, typename... RemainingArgTypes>
struct TaoRalApiFuncInvokerImpl<R, F, MemRefType<T, 0>, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    MemRefType<T, 0> memref;
    memref.basePtr = *(T**)(*args++);
    memref.data = *(T**)(*args++);
    memref.offset = *(int64_t*)(*args++);
    TaoRalApiFuncInvokerImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args, std::forward<ParsedArgs>(parsed_args)..., std::move(memref));
  }
};

template <typename R, typename F>
struct TaoRalApiFuncInvokerImpl<R, F> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    *(R*)*args = std::move(f(std::forward<ParsedArgs>(parsed_args)...));
  }
};

template <int N>
struct TupleAssign {
  template <typename... Rs>
  static inline void Invoke(const std::tuple<Rs...>& rs, void** args) {
    constexpr const int n = sizeof...(Rs) - N;
    using T0 = decltype(std::get<n>(rs));
    using T1 = typename std::remove_reference<T0>::type;
    using T2 = typename std::remove_const<T1>::type;
    *(T2*)(*args++) = std::get<n>(rs);
    TupleAssign<N - 1>::Invoke(rs, args);
  }
};

template <>
struct TupleAssign<0> {
  template <typename... Rs>
  static inline void Invoke(const std::tuple<Rs...>& rs, void** args) {}
};

template <typename... Rs, typename F>
struct TaoRalApiFuncInvokerImpl<std::tuple<Rs...>, F> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    TupleAssign<sizeof...(Rs)>::Invoke(
        f(std::forward<ParsedArgs>(parsed_args)...), args);
  }
};

template <typename F>
struct TaoRalApiFuncInvokerImpl<void, F> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    f(std::forward<ParsedArgs>(parsed_args)...);
  }
};

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct TaoRalApiFuncInvoker<Return (*)(Args...), impl_fn> {
  static void Invoke(void** args) {
    TaoRalApiFuncInvokerImpl<Return, Return (*)(Args...), Args...>::Invoke(
        impl_fn, args);
  }
};

class TaoRalApiRegistry {
 public:
  ~TaoRalApiRegistry();
  using api_func_t = tao::ral::api_func_t;
  static TaoRalApiRegistry& Global();
  // Registers a api_func identified by name `name`, and if `nickname` is not
  // exists in the registry, also register this api_func under `nickname`
  bool Register(const std::string& name, const std::string& nickname,
                api_func_t api_func);
  api_func_t Find(const std::string& name);

 private:
  TaoRalApiRegistry();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Macros used to define TAO_RAL apis.
#define TAO_RAL_API(name, device, ...) \
  TAO_RAL_API_UNIQ_HELPER(name, device, __COUNTER__, __VA_ARGS__)

#define TAO_RAL_API_UNIQ_HELPER(name, device, ctr, ...) \
  TAO_RAL_API_UNIQ(name, device, ctr, __VA_ARGS__)

#define TAO_RAL_API_UNIQ(name, device, ctr, ...)                               \
  static bool unused_ret_val_##ctr =                                           \
      ::tao::ral::TaoRalApiRegistry::Global().Register(                        \
          ::tao::ral::TaoRalApiFuncNameHelper<decltype(&__VA_ARGS__)>::Invoke( \
              std::string(name) + "___" + std::string(device)),                \
          std::string(name),                                                   \
          ::tao::ral::TaoRalApiFuncInvoker<decltype(&__VA_ARGS__),             \
                                           &__VA_ARGS__>::Invoke);

// =============================== DriverApiFuncInvokerHelper ==============

using api_func_t = TaoRalApiRegistry::api_func_t;

template <typename R, typename... ArgTypes>
struct DriverApiWrapperImpl;

template <typename T, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<void, T, RemainingArgTypes...> {
  static inline void Invoke(api_func_t f, std::vector<void*> args, T&& arg,
                            RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&arg));
    DriverApiWrapperImpl<void, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename R, typename T, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<R, T, RemainingArgTypes...> {
  static inline R Invoke(api_func_t f, std::vector<void*> args, T&& arg,
                         RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&arg));
    return DriverApiWrapperImpl<R, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename R, typename T, int N, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<R, MemRefType<T, N>, RemainingArgTypes...> {
  static inline R Invoke(api_func_t f, std::vector<void*> args,
                         MemRefType<T, N> memref,
                         RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&memref.basePtr));
    args.push_back(static_cast<void*>(&memref.data));
    args.push_back(static_cast<void*>(&memref.offset));
    for (int i = 0; i < N; ++i) {
      args.push_back(static_cast<void*>(&memref.sizes[i]));
    }
    for (int i = 0; i < N; ++i) {
      args.push_back(static_cast<void*>(&memref.strides[i]));
    }
    return DriverApiWrapperImpl<R, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename R, typename T, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<R, MemRefType<T, 0>, RemainingArgTypes...> {
  static inline R Invoke(api_func_t f, std::vector<void*> args,
                         MemRefType<T, 0> memref,
                         RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&memref.basePtr));
    args.push_back(static_cast<void*>(&memref.data));
    args.push_back(static_cast<void*>(&memref.offset));
    return DriverApiWrapperImpl<R, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename T, int N, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<void, MemRefType<T, N>, RemainingArgTypes...> {
  static inline void Invoke(api_func_t f, std::vector<void*> args,
                            MemRefType<T, N> memref,
                            RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&memref.basePtr));
    args.push_back(static_cast<void*>(&memref.data));
    args.push_back(static_cast<void*>(&memref.offset));
    for (int i = 0; i < N; ++i) {
      args.push_back(static_cast<void*>(&memref.sizes[i]));
    }
    for (int i = 0; i < N; ++i) {
      args.push_back(static_cast<void*>(&memref.strides[i]));
    }
    return DriverApiWrapperImpl<void, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename T, typename... RemainingArgTypes>
struct DriverApiWrapperImpl<void, MemRefType<T, 0>, RemainingArgTypes...> {
  static inline void Invoke(api_func_t f, std::vector<void*> args,
                            MemRefType<T, 0> memref,
                            RemainingArgTypes&&... remaining_args) {
    args.push_back(static_cast<void*>(&memref.basePtr));
    args.push_back(static_cast<void*>(&memref.data));
    args.push_back(static_cast<void*>(&memref.offset));
    return DriverApiWrapperImpl<void, RemainingArgTypes...>::Invoke(
        f, args, std::forward<RemainingArgTypes>(remaining_args)...);
  }
};

template <typename R>
struct DriverApiWrapperImpl<R> {
  static inline R Invoke(api_func_t f, std::vector<void*> args) {
    R r;
    args.push_back(&r);
    f(args.data());
    return r;
  }
};

template <>
struct DriverApiWrapperImpl<void> {
  static inline void Invoke(api_func_t f, std::vector<void*> args) {
    f(args.data());
  }
};

template <typename T>
struct DriverApiWrapper;

template <typename R, typename... Args>
struct DriverApiWrapper<std::function<R(Args...)>> {
  static inline std::function<R(Args...)> Wrapper(api_func_t func) {
    if (!func) return nullptr;
    return [func](Args&&... args) -> R {
      std::vector<void*> arg_ptrs;
      return DriverApiWrapperImpl<R, Args...>::Invoke(
          func, arg_ptrs, std::forward<Args>(args)...);
    };
  }
};

template <typename... Args>
struct DriverApiWrapper<std::function<void(Args...)>> {
  static inline std::function<void(Args...)> Wrapper(api_func_t func) {
    if (!func) return nullptr;
    return [func](Args&&... args) -> void {
      std::vector<void*> arg_ptrs;
      DriverApiWrapperImpl<void, Args...>::Invoke(func, arg_ptrs,
                                                  std::forward<Args>(args)...);
    };
  }
};

#define TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(wrapper, api_func) \
  wrapper = DriverApiWrapper<decltype(wrapper)>::Wrapper(api_func);

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_HELPER_H_
