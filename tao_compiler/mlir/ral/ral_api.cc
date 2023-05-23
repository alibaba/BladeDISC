//===- tao_ral.cc ----------------------===//
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

#include "mlir/ral/ral_api.h"

#include <iostream>

#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_logging.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* const kMlirLoweredEntry = "main";

void tao_ral_call_impl(tao_ral_context_t ctx, void* name, void** args) {
  static int count = 0;
  TAO_VLOG(1) << "tao_ral_call is called with ctx = " << ctx;
  TAO_VLOG(1) << "ral call count #" << count++ << ": " << (const char*)name;
  auto typed_ctx = static_cast<tao::ral::ExecutionContext*>(ctx);
  typed_ctx->getContext()->call((const char*)name, args);
}

tao_ral_status_t tao_ral_last_error(tao_ral_context_t ctx,
                                    const char** err_msg) {
  TAO_VLOG(1) << "tao_ral_last_error is called with ctx = " << ctx;
  auto typed_ctx = static_cast<tao::ral::Context*>(ctx);
  return typed_ctx->getLastError(err_msg);
}

void tao_ral_clear_error(tao_ral_context_t ctx) {
  TAO_VLOG(1) << "tao_ral_clear_error is called with ctx = " << ctx;
  auto typed_ctx = static_cast<tao::ral::Context*>(ctx);
  return typed_ctx->clearError();
}

#ifdef __cplusplus
}
#endif
