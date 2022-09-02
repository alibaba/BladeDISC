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

#include "mlir/mhlo/builder/activation.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/broadcast.h"
#include "mlir/mhlo/builder/element_wise_binary.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {

// Converts Sigmoid op to HLO ops computing sigmoid with the following formula:
//     sigmoid(x) = 1.0 / (1.0 + exp(-x))
// Ref:
// https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid

mlir::Value BuildSigmoid(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& mlir_val) {
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(mlir_val);
  auto hlo_one = builder
                     .create<mlir::mhlo::ConstantOp>(
                         loc, builder.getFloatAttr(elem_type, 1.0))
                     .getResult();
  hlo_one =
      mlir::mhlo::BuildBroadcastScalarAsTensor(builder, loc, hlo_one, mlir_val);
  auto neg_val = builder.create<mlir::mhlo::NegOp>(loc, mlir_val);
  auto exp_val = builder.create<mlir::mhlo::ExpOp>(loc, neg_val);
  auto divisor = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, hlo_one, exp_val, elem_type,
      /* no_implicit_broadcast */ true);
  auto result = BuildMlirBinaryOp<mlir::chlo::BroadcastDivOp>(
      builder, loc, hlo_one, divisor, elem_type,
      /* no_implicit_broadcast */ true);
  return result;
}

}  // namespace mhlo
}  // namespace mlir
