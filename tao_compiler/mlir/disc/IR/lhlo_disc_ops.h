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

// This file defines the operations used in the mhlo_disc dialect.

#ifndef MLIR_DISC_IR_LHLO_DISC_OPS_H_
#define MLIR_DISC_IR_LHLO_DISC_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

namespace lmhlo_disc {

class LmhloDiscDialect : public Dialect {
 public:
  explicit LmhloDiscDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "lmhlo_disc"; }
};

}  // end namespace lmhlo_disc
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h.inc"

#endif  //  MLIR_DISC_IR_LHLO_DISC_OPS_H_
