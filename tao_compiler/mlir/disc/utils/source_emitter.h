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

#ifndef MLIR_DISC_UTILS_SOURCE_EMITTER
#define MLIR_DISC_UTILS_SOURCE_EMITTER

#include <string>

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace disc_ral {

class SourceEmitterCUDA {
 public:
  using ValueNameBinding = llvm::DenseMap<Value, std::string>;

 public:
  static bool isBroadcastOnScalarOrSplatConstant(Operation* op);
  static bool isSupportedOp(Operation* op);

  llvm::Optional<std::string> EmitOp(Operation* op, ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseUnaryOp(Operation* op,
                                                  ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseBinaryOp(Operation* op,
                                                   ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseTernaryOp(Operation* op,
                                                    ValueNameBinding& binding);

  llvm::Optional<std::string> EmitScalarOrSplatConstantOp(
      Operation* op, ValueNameBinding& binding);

  llvm::Optional<std::string> EmitBroadcastOfScalarOrSplatConstantOp(
      Operation* op, ValueNameBinding& binding);

  llvm::Optional<std::string> EmitDynamicReshapeOp(Operation* op,
                                                   ValueNameBinding& binding);
  llvm::Optional<std::string> EmitTransposeOp(Operation* op,
                                              ValueNameBinding& binding);

  void bindValueNames(const SmallVectorImpl<Value>& inputs,
                      const SmallVectorImpl<std::string>& names,
                      ValueNameBinding& binding);

 private:
  std::unordered_map<std::string, int32_t> existing_names_;

 private:
  std::string EmitUniqueName(llvm::StringRef op_str);
  llvm::Optional<std::string> EmitScalarOrSplatConstantExpression(
      lmhlo::ConstantOp constant);
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_UTILS_SOURCE_EMITTER