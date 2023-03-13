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

#include "mlir/disc/IR/rng_uniform_custom_call_op.h"

#include <atomic>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/disc/IR/custom_call_base.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

using ::mlir::mhlo_disc::RngUniformBackendConfig;

namespace llvm {
namespace json {

bool fromJSON(const llvm::json::Value& value,
              RngUniformBackendConfig& rng_uniform_backend_config,
              llvm::json::Path path) {
  ObjectMapper o(value, path);
  return o && o.map("seed", rng_uniform_backend_config.seed) &&
         o.map("seed2", rng_uniform_backend_config.seed2);
}

llvm::json::Value toJSON(
    const RngUniformBackendConfig& rng_uniform_backend_config) {
  return llvm::json::Value(Object{{"seed", rng_uniform_backend_config.seed},
                                  {"seed2", rng_uniform_backend_config.seed2}});
}

}  // namespace json
}  // namespace llvm

namespace mlir {
namespace mhlo_disc {

template <>
LogicalResult reifyReturnTypeShapesImpl<RngUniformBackendConfig>(
    CustomCallOp op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  Value operand = operands[2];
  auto operand_type = operand.getType().dyn_cast<ShapedType>();
  if (!operand_type || !operand_type.hasStaticShape() ||
      operand_type.getRank() != 1) {
    return op->emitOpError() << "the third operand of rng uniform op should be "
                                "rank-1 tensor with static shape";
  }
  Location loc = op.getLoc();
  SmallVector<Value, 4> shape_values;
  for (int i = 0; i < operand_type.getShape()[0]; ++i) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
    SmallVector<Value, 1> ivs(1, offset);
    Value dim_size = builder.create<tensor::ExtractOp>(loc, operand, ivs);
    dim_size = disc_ral::mayConvertToIndexType(dim_size, &builder, loc);
    shape_values.push_back(dim_size);
  }
  Value result_shape =
      builder.create<tensor::FromElementsOp>(loc, shape_values);
  reifiedReturnShapes.push_back(result_shape);
  return success();
}

}  // namespace mhlo_disc
namespace lmhlo_disc {

int64_t GetRngUniqueId() {
  static std::atomic<int64_t> id{0};
  return id++;
}

template <>
LogicalResult lowerToLibraryCallImpl<RngUniformBackendConfig>(
    CustomCallOp op, PatternRewriter& rewriter, Value ctx,
    Value stream_handle) {
  bool on_gpu = false;
  SmallVector<Value, 4> newOperands{stream_handle};
  for (Value operand : op.getOperands()) {
    newOperands.push_back(operand);
  }
  // seed & seed2
  llvm::Expected<RngUniformBackendConfig> backend_config =
      llvm::json::parse<RngUniformBackendConfig>(
          op.getBackendConfig().cast<StringAttr>());
  if (auto e = backend_config.takeError()) {
    return op.emitOpError()
           << "Problem with parsing rng_uniform backend_config: "
           << llvm::toString(std::move(e));
  }
  newOperands.push_back(
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), GetRngUniqueId(), 64));
  newOperands.push_back(rewriter.create<arith::ConstantIntOp>(
      op.getLoc(), backend_config->seed, 64));
  newOperands.push_back(rewriter.create<arith::ConstantIntOp>(
      op.getLoc(), backend_config->seed2, 64));
  on_gpu = placement_utils::isGpuMemRef(op->getOperand(3));
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, ctx, newOperands, "ral_gpu_rng_uniform",
      /*has_side_effect*/ false,
      /*backend_config*/ on_gpu ? "gpu" : "cpu");
  return success();
}

}  // namespace lmhlo_disc

REGISTER_CUSTOM_CALL(
    "rng_uniform",
    mhlo_disc::reifyReturnTypeShapesImpl<RngUniformBackendConfig>,
    lmhlo_disc::lowerToLibraryCallImpl<RngUniformBackendConfig>);

}  // namespace mlir
