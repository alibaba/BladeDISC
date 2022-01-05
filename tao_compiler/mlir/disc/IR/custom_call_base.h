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

// This file defines the basic macros for custom calls.

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CUSTOM_CALL_BASE_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CUSTOM_CALL_BASE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class OpBuilder;
class PatternRewriter;
class Value;
class ValueRange;
template <typename T> class SmallVectorImpl;

namespace lmhlo_disc {
class CustomCallOp;
}

namespace mhlo_disc {
class CustomCallOp;

class CustomCallRegistry {
public:
  ~CustomCallRegistry();
  using reify_shapes_func_t = std::function<LogicalResult(
      CustomCallOp op, OpBuilder &builder, ValueRange operands,
      SmallVectorImpl<Value> &reifiedReturnShapes)>;
  using lower_to_library_call_func_t = std::function<LogicalResult(
      lmhlo_disc::CustomCallOp op, PatternRewriter &rewriter, Value ctx,
      Value stream_handle)>;
  static CustomCallRegistry &Global();
  bool Register(const std::string &name, reify_shapes_func_t reify_shapes_func,
                lower_to_library_call_func_t lower_to_library_call_func);
  reify_shapes_func_t FindReifyShapesFunc(const std::string &name);
  lower_to_library_call_func_t
  FindLowerToLibraryCallFunc(const std::string &name);

private:
  CustomCallRegistry();
  class Impl;
  std::unique_ptr<Impl> impl_;
};

template <typename BackendConfig>
LogicalResult
reifyReturnTypeShapesImpl(CustomCallOp op, OpBuilder &builder,
                          ValueRange operands,
                          SmallVectorImpl<Value> &reifiedReturnShapes);

} // end namespace mhlo_disc

#define REGISTER_CUSTOM_CALL(name, reify_shapes_func,                          \
                             lower_to_library_call_func)                       \
  static bool unused_ret_val_##ctr =                                           \
      ::mlir::mhlo_disc::CustomCallRegistry::Global().Register(                \
          std::string(name), reify_shapes_func, lower_to_library_call_func);

namespace lmhlo_disc {
class CustomCallOp;

template <typename BackendConfig>
LogicalResult lowerToLibraryCallImpl(CustomCallOp op, PatternRewriter &rewriter,
                                     Value ctx, Value stream_handle);

} // end namespace lmhlo_disc
} // end namespace mlir

#endif // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CUSTOM_CALL_BASE_H_
