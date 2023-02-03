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

#include "mlir/disc/IR/topk_custom_call_op.h"

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/disc/IR/custom_call_base.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

using ::mlir::mhlo_disc::TopKBackendConfig;

namespace llvm {
namespace json {

bool fromJSON(const llvm::json::Value& value,
              TopKBackendConfig& topk_backend_config, llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && o.map("dimension", topk_backend_config.dimension);
}

llvm::json::Value toJSON(const TopKBackendConfig& topk_backend_config) {
  return llvm::json::Value(
      Object{{"dimension", topk_backend_config.dimension}});
}

}  // namespace json
}  // namespace llvm

namespace mlir {
namespace mhlo_disc {

template <>
LogicalResult reifyReturnTypeShapesImpl<TopKBackendConfig>(
    CustomCallOp op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  llvm::Expected<TopKBackendConfig> backend_config =
      llvm::json::parse<TopKBackendConfig>(
          op.getBackendConfig().cast<StringAttr>());
  int64_t dimension = backend_config->dimension;

  Value keys_operand = operands[0];
  // TopK CustomCall input operands: keys, values, k
  if (operands.size() != 3) {
    return op.emitOpError()
           << "TopK CustomCallOp requires 3 input operands: keys, values, k";
  }
  ShapedType operand_type = keys_operand.getType().dyn_cast<RankedTensorType>();
  if (!operand_type) {
    return op.emitOpError()
           << "keys of TopK CustomCallOp must be ranked tensor";
  }
  auto rank = operand_type.getRank();
  if (rank > 2) {
    return op.emitOpError()
           << "keys of TopK CustomCallOp currently only supports rank <= 2";
  }
  auto value_operand_type = operands[1].getType().dyn_cast<ShapedType>();
  if ((!value_operand_type) || (value_operand_type.getRank() != rank)) {
    return op.emitOpError() << "the rank of key/value operand of TopK "
                               "CustomCall must be the same";
  }
  auto k_operand = operands[2];
  auto k_operand_type = k_operand.getType().dyn_cast<RankedTensorType>();
  if (!k_operand_type || k_operand_type.getRank() != 0) {
    return op.emitOpError()
           << "operand k of TopK CustomCall must be a scalar tensor ranked 0";
  }
  // Support of negative number definition.
  dimension = (dimension + rank) % rank;
  if (dimension != (rank - 1)) {
    return op.emitOpError() << "TopK CustomCall currently only supporting "
                               "sorting at last dimension.";
  }

  auto loc = op->getLoc();
  Value k_value = builder.create<tensor::ExtractOp>(loc, k_operand);
  k_value = disc_ral::mayConvertToIndexType(k_value, &builder, loc);
  SmallVector<Value> shape_values;
  shape_values.reserve(rank);
  for (auto element : llvm::enumerate(operand_type.getShape())) {
    int64_t idx = element.index();
    if (idx == dimension) {
      auto neg_one = builder.create<arith::ConstantIndexOp>(loc, -1);
      auto cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                k_value, neg_one);
      Value true_br_value =
          builder.create<tensor::DimOp>(loc, keys_operand, idx);
      auto select = builder.create<mlir::arith::SelectOp>(
          loc, cond, true_br_value, k_value);
      shape_values.push_back(select);
    } else {
      Value dim = builder.create<tensor::DimOp>(loc, keys_operand, idx);
      shape_values.push_back(dim);
    }
  }
  Value result_shape =
      builder.create<tensor::FromElementsOp>(loc, shape_values);
  // Same shapes for two outputs
  reifiedReturnShapes.push_back(result_shape);
  reifiedReturnShapes.push_back(result_shape);
  return success();
}

}  // namespace mhlo_disc
namespace lmhlo_disc {

template <>
LogicalResult lowerToLibraryCallImpl<TopKBackendConfig>(
    CustomCallOp op, PatternRewriter& rewriter, Value ctx,
    Value stream_handle) {
  bool on_gpu = false;
  SmallVector<Value, 4> newOperands{stream_handle};
  for (Value operand : op.getOperands()) {
    newOperands.push_back(operand);
  }
  // dimension
  llvm::Expected<TopKBackendConfig> backend_config =
      llvm::json::parse<TopKBackendConfig>(
          op.getBackendConfig().cast<StringAttr>());
  if (auto e = backend_config.takeError()) {
    return op.emitOpError() << "Problem with parsing topk backend_config: "
                            << llvm::toString(std::move(e));
  }
  newOperands.push_back(rewriter.create<arith::ConstantIntOp>(
      op.getLoc(), backend_config->dimension, 64));
  // is_ascending
  newOperands.push_back(
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 1));
  on_gpu = placement_utils::isGpuMemRef(op->getOperand(3));
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, ctx, newOperands, "ral_dsort",
      /*has_side_effect*/ false,
      /*backend_config*/ on_gpu ? "gpu" : "cpu");
  return success();
}

}  // namespace lmhlo_disc

REGISTER_CUSTOM_CALL("topk",
                     mhlo_disc::reifyReturnTypeShapesImpl<TopKBackendConfig>,
                     lmhlo_disc::lowerToLibraryCallImpl<TopKBackendConfig>);

}  // namespace mlir
