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

// This file defines topk custom call.

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_RNG_UNIFORM_CUSTOM_CALL_OP_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_RNG_UNIFORM_CUSTOM_CALL_OP_H_

#include "llvm/Support/JSON.h"

namespace mlir {

class Value;
class OpBuilder;
class ValueRange;

namespace mhlo_disc {

using llvm::json::Object;
using llvm::json::ObjectMapper;

class CustomCallOp;

struct RngUniformBackendConfig {
  RngUniformBackendConfig() : seed(1), seed2(2) {}
  RngUniformBackendConfig(int64_t seed, int64_t seed2)
      : seed(seed), seed2(seed2) {}
  int64_t seed;
  int64_t seed2;
};

} // namespace mhlo_disc
} // namespace mlir

namespace llvm {
namespace json {

bool fromJSON(
    const llvm::json::Value &value,
    ::mlir::mhlo_disc::RngUniformBackendConfig &rng_uniform_backend_config,
    llvm::json::Path path);
llvm::json::Value toJSON(const ::mlir::mhlo_disc::RngUniformBackendConfig
                             &rng_uniform_backend_config);

} // namespace json
} // namespace llvm

#endif //   TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_RNG_UNIFORM_CUSTOM_CALL_OP_H_
