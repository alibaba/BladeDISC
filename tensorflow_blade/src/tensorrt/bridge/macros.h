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

#ifndef __COMMON_MACROS_H__
#define __COMMON_MACROS_H__

#define TorchAddonsDeclNewFlag(T, Name) \
  T Set##Name(T);                       \
  T& Get##Name();

#define TorchAddonsDefNewFlag(T, Name) \
  T& Get##Name() {                     \
    thread_local T flag;               \
    return flag;                       \
  }                                    \
  T Set##Name(T flag) {                \
    T old_flag = Get##Name();          \
    Get##Name() = flag;                \
    return old_flag;                   \
  }

#endif  //__COMMON_MACROS_H__
