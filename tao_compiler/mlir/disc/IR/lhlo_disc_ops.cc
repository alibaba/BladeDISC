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

// This file defines the operations used in the LMHLO DISC dialect.

#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"

#include <unordered_set>

namespace mlir {
namespace lmhlo_disc {

//===----------------------------------------------------------------------===//
// lmhlo disc Dialect Constructor
//===----------------------------------------------------------------------===//

LmhloDiscDialect::LmhloDiscDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<LmhloDiscDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.cc.inc"

      >();
  context->loadDialect<memref::MemRefDialect>();
}

//===----------------------------------------------------------------------===//
// CustomCallOp.
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CustomCallOp op) {
  if (op.target_arg_mapping()) {
    lmhlo::CustomCallTargetArgMapping mapping = *op.target_arg_mapping();
    auto verify_mapping = [&](int64_t target_num, size_t op_num,
                              ArrayAttr mapping,
                              StringRef kind) -> LogicalResult {
      if (target_num < op_num)
        return op.emitOpError("number of target " + kind + " (")
               << target_num << ") cannot be less than the number of " << kind
               << "(" << op_num << ") for the operation";

      if (mapping.size() != op_num)
        return op.emitOpError("number of entries in the mapping for " + kind +
                              " (")
               << mapping.size() << ") should match the number of " << kind
               << " for the operation (" << op_num << ")";

      std::unordered_set<int64_t> entries;
      // Each entry in the mapping should be < target_num and an entry cannot
      // appear more than once.
      for (Attribute entry : mapping) {
        int64_t int_entry = entry.cast<IntegerAttr>().getInt();
        // ODS verification will ensure that these entries are integers.
        if (!entries.insert(int_entry).second)
          return op.emitOpError("entry ")
                 << int_entry
                 << " cannot appear more than once in the mapping for " << kind;
        if (int_entry < 0 || int_entry >= target_num)
          return op.emitOpError(
                     "entries in mapping for " + kind +
                     " must be >= 0 and less than target's number of " + kind +
                     " (")
                 << target_num << ")";
      }
      return success();
    };
    if (failed(verify_mapping(mapping.num_args().getInt(), op.args().size(),
                              mapping.args_to_target_args(), "args")) ||
        failed(verify_mapping(mapping.num_results().getInt(),
                              op.output().size(),
                              mapping.results_to_target_results(), "results")))
      return failure();
  }
  return success();
}

}  // namespace lmhlo_disc
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.cc.inc"
