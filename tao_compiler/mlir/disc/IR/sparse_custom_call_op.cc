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

// This file defines sparse gemm related custom calls.
#include "lhlo/IR/lhlo_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/disc/IR/custom_call_base.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace mhlo_disc {

LogicalResult reifyReturnTypeShapesSparseGemmImpl(
    CustomCallOp op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  if (op->getNumOperands() < 2 || op->getNumResults() != 1)
    return op->emitError() << "mismatch #operands or #results\n";

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);

  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy) return op->emitError() << "not support unranked type\n";

  Location loc = op.getLoc();
  auto config = op.getBackendConfig().cast<DictionaryAttr>();
  int64_t lhs_contracting_dimensions =
      config.getAs<IntegerAttr>("lhs_contracting_dimensions").getInt();
  int64_t rhs_contracting_dimensions =
      config.getAs<IntegerAttr>("rhs_contracting_dimensions").getInt();

  Value m = builder.create<tensor::DimOp>(
      loc, lhs, lhs_contracting_dimensions == 1 ? 0 : 1);
  Value n = builder.create<tensor::DimOp>(
      loc, rhs, rhs_contracting_dimensions == 0 ? 1 : 0);

  SmallVector<Value> shape_values{n, m};
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, shape_values);
  reifiedReturnShapes.push_back(output_shape);
  return success();
}

}  // namespace mhlo_disc
namespace lmhlo_disc {

LogicalResult lowerToLibraryCallSpargeGemmImpl(CustomCallOp op,
                                               PatternRewriter& rewriter,
                                               Value ctx, Value stream_handle) {
  if (op->getNumOperands() < 3 || op->getNumResults() != 0)
    return op->emitError() << "mismatch #operands or #results\n";
  SmallVector<Value> newOperands{stream_handle};

  for (int i = 0; i < op->getNumOperands(); ++i) {
    newOperands.push_back(op.getOperand(i));
  }

  Location loc = op.getLoc();
  auto config = op.getBackendConfig().cast<DictionaryAttr>();
  int64_t lhs_contracting_dimensions =
      config.getAs<IntegerAttr>("lhs_contracting_dimensions").getInt();
  int64_t rhs_contracting_dimensions =
      config.getAs<IntegerAttr>("rhs_contracting_dimensions").getInt();
  Type field_type = rewriter.getI32Type();

  Value input_cd = rewriter.create<arith::ConstantIntOp>(
      loc, lhs_contracting_dimensions, field_type);
  Value kernel_cd = rewriter.create<arith::ConstantIntOp>(
      loc, rhs_contracting_dimensions, field_type);

  newOperands.push_back(input_cd);
  newOperands.push_back(kernel_cd);

  bool on_gpu = placement_utils::isGpuMemRef(op->getOperand(0));
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, ctx, newOperands, "sparse_gemm", false,
      on_gpu ? "gpu" : "cpu");
  return success();
}

}  // namespace lmhlo_disc

REGISTER_CUSTOM_CALL("sparse_gemm",
                     mhlo_disc::reifyReturnTypeShapesSparseGemmImpl,
                     lmhlo_disc::lowerToLibraryCallSpargeGemmImpl);

}  // namespace mlir
