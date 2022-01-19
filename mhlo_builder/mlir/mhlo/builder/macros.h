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

#pragma once
#include <sstream>
#include <stdexcept>

namespace mlir {
namespace mhlo {
// Return x if it is non-empty; otherwise return y.
inline std::string if_empty_then(std::string x, std::string y) {
  if (x.empty()) {
    return y;
  } else {
    return x;
  }
}

template <typename T>
void ssprint(std::stringstream& ss, T t) {
  ss << t;
}

template <typename T, typename... Args>
void ssprint(std::stringstream& ss, T t,
             Args... args)  // recursive variadic function
{
  ss << t;
  ssprint(ss, args...);
}

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define MHLO_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define MHLO_UNLIKELY(expr) (expr)
#endif

#define MHLO_THROW_ERROR(msg) throw std::runtime_error(msg)

#define MHLO_CHECK(cond, ...)                                           \
  if (MHLO_UNLIKELY(!(cond))) {                                         \
    std::stringstream msg_ss;                                           \
    ssprint(msg_ss, __func__, __FILE__);                                \
    ssprint(msg_ss, static_cast<uint32_t>(__LINE__), __VA_ARGS__);      \
    MHLO_THROW_ERROR(if_empty_then(                                     \
        msg_ss.str(), "Expected " #cond " to be true, but got false.  " \
                      "(Could this error message be improved?  If so, " \
                      "please report an enhancement request.)"));       \
  }
}  // namespace mhlo
}  // namespace mlir
