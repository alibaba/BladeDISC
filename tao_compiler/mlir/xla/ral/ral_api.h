//===- ral_api.h ----------------------===//
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

#ifndef RAL_RAL_API_H_
#define RAL_RAL_API_H_

// RAL (Runtime Abstraction Layer) API, which is the interfaces between
// the compiler and runtime.

// RAL API statifes:
//   - the first argument is always a context handle
//   - error is fetched by an additional call to `tao_ral_get_last_error`
//   - RAL API is initialization free, resource lifetime is automatically
//     managed by the context.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const char* const kMlirLoweredEntry;

// ========================= Type Definitions ===================
// context handle
typedef void* tao_ral_context_t;

// integer status type, used for error checking
// value zero is always ok, otherwise is failed.
typedef int32_t tao_ral_status_t;

// ========================== API Definitions ====================

// Custom call abstraction.
// Calls a pre-registered custom kernel identified by `name` and
// forward args to the kernel.

// TODO: Ideally we should hide all other apis using this api and make the
// ABI clean and stable. This is even more important if we support
// multi-device in the future (since device is pluggable, one evironment
// has suport for device type `A` may not also have support for device
// type `B`).

void tao_ral_call_impl(tao_ral_context_t, void* name, void** args);

// Returns the status since the last api call.
// When error occurs, error msg is stored into `err_msg` if it's
// not null. `err_msg` is empty is status is ok.
tao_ral_status_t tao_ral_last_error(tao_ral_context_t, const char** err_msg);

// Clears if is in error status.
void tao_ral_clear_error(tao_ral_context_t);

#ifdef __cplusplus
}
#endif

#endif  // RAL_RAL_API_H_
