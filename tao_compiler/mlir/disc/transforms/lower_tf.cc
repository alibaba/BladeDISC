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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                          // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"           // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/rng_uniform_custom_call_op.h"
#include "tensorflow/compiler/mlir/disc/IR/topk_custom_call_op.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "disc-lower-tf"

namespace mlir {
namespace disc_ral {
namespace {

ValueRange PackRandomUniformInputs(Value lb, Value ub, Value shape) {
  return {lb, ub, shape};
}

StringAttr PackRandomUniformBackendConfig(IntegerAttr seed, IntegerAttr seed2,
                                          PatternRewriter* rewriter) {
  mhlo_disc::RngUniformBackendConfig config(seed.getValue().getSExtValue(),
                                            seed2.getValue().getSExtValue());
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << ::llvm::json::toJSON(config);
  return rewriter->getStringAttr(ostream.str());
}

// Prepare TF operations in functions for subsequent legalization.
struct PrepareTFPass : public DiscLowerTfPassBase<PrepareTFPass> {
  using DiscLowerTfPassBase<PrepareTFPass>::DiscLowerTfPassBase;

  // TODO: move to td file
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mhlo_disc::MhloDiscDialect>();
  }

  void runOnOperation() override;
};

// Converts a tf.SqueezeOp to xla_hlo.ReshapeOp
// SqueezeOp with empty squeeze_dims is a dynamic rank op and will not be
// supported
class ConvertSqueezeOpDynamic : public OpRewritePattern<TF::SqueezeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::SqueezeOp op,
                                PatternRewriter& rewriter) const final {
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getType(),
                                                   op.input());
    } else {
      Location loc = op.getLoc();
      Value input = op.input();
      auto input_ty = input.getType().dyn_cast<RankedTensorType>();
      if (!input_ty) {
        return failure();
      }
      int64_t input_rank = input_ty.getRank();
      auto squeeze_dims_attr = op.squeeze_dims();
      int64_t squeeze_dims_attr_size = squeeze_dims_attr.size();
      if (squeeze_dims_attr_size == 0) {
        return failure();
      }
      llvm::SetVector<int64_t> squeeze_dims;
      for (int64_t i = 0; i < squeeze_dims_attr_size; ++i) {
        squeeze_dims.insert(squeeze_dims_attr[i].cast<IntegerAttr>().getInt());
      }
      SmallVector<Value, 4> shape_values;
      int64_t output_rank = input_rank - squeeze_dims_attr_size;
      shape_values.reserve(output_rank);
      for (int64_t i = 0; i < input_rank; ++i) {
        if (squeeze_dims.count(i)) {
          continue;
        }
        auto dim_size = input_ty.getDimSize(i);
        if (dim_size == ShapedType::kDynamicSize) {
          shape_values.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
        } else {
          shape_values.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, dim_size));
        }
      }
      Value new_shape =
          rewriter.create<tensor::FromElementsOp>(loc, shape_values);
      rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
          op, result_ty, op.input(), new_shape);
    }
    return success();
  }
};

// Converts a tf.TopKV2Op to topk custom_call
class ConvertTopKV2OpDynamic : public OpRewritePattern<TF::TopKV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::TopKV2Op op,
                                PatternRewriter& rewriter) const final {
    auto k_type = op.k().getType().dyn_cast<RankedTensorType>();
    if (!k_type || (k_type.getElementType() != rewriter.getIntegerType(32)) ||
        (k_type.getRank() != 0)) {
      return op.emitOpError() << "TopKV2 requires 0D scalar tensor k";
    }

    auto input_type = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_type) {
      return op.emitOpError() << "TopKV2 requires input to be ranked tensor";
    }
    int64_t input_rank = input_type.getRank();
    int64_t last_dim_index = input_rank - 1;

    // Create an Itoa op for indices.
    // TODO(zk): if we always choose to use tf kernel implementation, then
    // this iota is redundant and should be removed.
    Type iota_type = RankedTensorType::get(input_type.getShape(),
                                           rewriter.getIntegerType(32));
    SmallVector<Value, 4> iota_shape_values;
    iota_shape_values.reserve(input_rank);
    for (int64_t idx = 0; idx < input_rank; ++idx) {
      Value dim = rewriter.create<tensor::DimOp>(op.getLoc(), op.input(), idx);
      iota_shape_values.push_back(dim);
    }
    Value iota_shape =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), iota_shape_values);
    Value iota_op = rewriter.create<mhlo::DynamicIotaOp>(
        op.getLoc(), iota_type, iota_shape,
        rewriter.getI64IntegerAttr(last_dim_index));

    // Create the topk custom call. It takes 3 inputs: keys, values and scalar
    // k. And generates two output: sorted_topk_keys, sorted_topk_values
    mhlo_disc::TopKBackendConfig backend_config(last_dim_index);
    std::string str;
    llvm::raw_string_ostream ostream(str);
    ostream << ::llvm::json::toJSON(backend_config);
    auto topk_custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
        op.getLoc(),
        TypeRange{op.getResult(0).getType(), op.getResult(1).getType()},
        ValueRange{op.input(), iota_op, op.k()},
        /*call_target_name*/ "topk",
        /*has_side_effect*/ false,
        /*backend_config*/ ostream.str());
    rewriter.replaceOp(op, {topk_custom_call_op.getResult(0),
                            topk_custom_call_op.getResult(1)});
    return success();
  }
};

// Convert a tf.RandomUniformOp to random_uniform custom_call
class ConvertUniformOp : public OpRewritePattern<TF::RandomUniformOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::RandomUniformOp op,
                                PatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    Value zero = rewriter.create<mhlo::ConstOp>(
        loc, rewriter.getFloatAttr(op.dtype(), 0.0));
    Value one = rewriter.create<mhlo::ConstOp>(
        loc, rewriter.getFloatAttr(op.dtype(), 1.0));
    auto cfg = PackRandomUniformBackendConfig(
        rewriter.getIntegerAttr(op.dtype(), op.seed()),
        rewriter.getIntegerAttr(op.dtype(), op.seed2()), &rewriter);
    auto custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
        loc, TypeRange{op.getResult().getType()},
        ValueRange{zero, one, op.shape()},
        /*custom call target*/ rewriter.getStringAttr("rng_uniform"),
        /*has_side_effect*/ rewriter.getBoolAttr(false),
        /*backend_config*/ cfg);
    rewriter.replaceOp(op, {custom_call_op.getResult(0)});
    return success();
  }
};

#include "tensorflow/compiler/mlir/disc/transforms/lower_tf.inc"

void PrepareTFPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  FuncOp func = getOperation();
  populateWithGenerated(patterns);
  patterns.insert<ConvertSqueezeOpDynamic, ConvertTopKV2OpDynamic,
                  ConvertUniformOp>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Lower some tf ops before tf2mhlo lowering.
std::unique_ptr<OperationPass<FuncOp>> createDiscLowerTfPass() {
  return std::make_unique<PrepareTFPass>();
}

}  // namespace disc_ral
}  // namespace mlir
