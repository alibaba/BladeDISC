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

#include "mlir/disc/IR/disc_tf_additional_ops.h"

namespace mlir {
namespace TF {

LogicalResult DiscFakeQuantOp::verify() {
  auto q_min = static_cast<int64_t>(quant_min());
  auto q_max = static_cast<int64_t>(quant_max());
  if (q_min >= q_max) {
    return emitOpError("quant_min (")
           << q_min << ") must be less than quant_max (" << q_max << ")";
  }

  int64_t expect_min = use_signed() ? (1 << (num_bits() - 1)) * -1 : 0;
  int64_t expect_max =
      use_signed() ? (1 << (num_bits() - 1)) - 1 : (1 << num_bits()) - 1;

  if (q_min < expect_min) {
    return emitOpError("quant_min must not be less than ")
           << expect_min << " under " << num_bits()
           << " bits and signed=" << use_signed() << ", but got: " << q_min;
  }
  if (q_max > expect_max) {
    return emitOpError("quant_max must not be greater than ")
           << expect_max << " under " << num_bits()
           << " bits and signed=" << use_signed() << ", but got: " << q_max;
  }
  if (axis().size() > 1) {
    return emitOpError(
               "axis must be empty (per-tensor) or a single element array "
               "(per-channel), but got: ")
           << axis();
  }
  return success();
}

namespace {

void DiscTFOpsRegisterHook(TensorFlowDialect& dialect) {
  dialect.addOperations<
#define GET_OP_LIST
#include "mlir/disc/IR/disc_tf_additional_ops.cc.inc"
      >();
}

int RegisterOnce() {
  TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(DiscTFOpsRegisterHook);
  return 0;
}

auto register_once = RegisterOnce();

}  // namespace
}  // namespace TF
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/IR/disc_tf_additional_ops.cc.inc"
