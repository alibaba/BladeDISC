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

// This file implements the logic to propagate some known shape information.
// The basic flow is shown as below:
//   loop until converged:
//     stage #1: rewrite locally (pattern based)
//     - run applyPatternsAndFoldGreedily(...), where patterns from:
//         MhloOpShapeRefinerPattern #A/../#Z
//         TensorOrShapeOpsRefinerPatterns/Other patterns (e.g. const folding)
//     - example mhlo shape refiner pattern:
//       original:
//         %1 = "mhlo.XXXOp"(%0) : (tensor<?x10xf32>) -> tensor<?x?xf32>
//       to:
//         %1 = "mhlo.XXXOp"(%0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
//         // convert to original shape to remain valid IR and rely on the
//         second
//         // stage to propagate such information globally.
//         %2 = tensor.cast %1 : tensor<?x10xf32> to tensor<?x?xf32>
//
//     stage #2: propagate shape information globally, examples are:
//      convert from:
//        func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<10xf32>) ->
//        tensor<?xf32> {
//          %0 = tensor.cast %arg1 : tensor<10xf32> to tensor<?xf32>
//          %1 = "mhlo.add"(%arg0, %0) : (tensor<?xf32>, tensor<?xf32>) ->
//          tensor<?xf32> return %1 : tensor<?xf32>
//        }
//      to:
//        func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) ->
//        tensor<10xf32> {
//          %1 = "mhlo.add"(%arg0, %0) : (tensor<10xf32>, tensor<10xf32>) ->
//          tensor<10xf32> return %1 : tensor<10xf32>
//        }
//     stage #3: apply symbolic shape optimizations, examples are:
//      - broadcast optimizations:
//        - %output = broadcast(%input, %shape) -> %output = %input if %input
//        and %output have the same shape.

#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {
namespace {

// convert:
//   %shape = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
//   %dim_size = tensor.extract %shape[%c0] : tensor<2xindex>
// to:
//   %dim_size = tensor.dim %0, %c0 : tensor<2xindex>
struct ExtractFromExtentTensorCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto shape_of_op = op.getTensor().getDefiningOp<shape::ShapeOfOp>();
    if (!shape_of_op) return failure();
    Value index = op.getIndices().front();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, shape_of_op.getArg(), index);
    return success();
  }
};

// Canonicalizes
// %c4_i32 = constant 4 : i32
// %shape = tensor.from_elements %0, %c4_i32 : tensor<2xi32>
// %1 = "mhlo.dynamic_reshape"(%tensor, %shape)  -> tensor<?x?xf32>
//
// into:
//
// %c4_i32 = constant 4 : i32
// %shape = tensor.from_elements %0, %c4_i32 : tensor<2xi32>
// %t = "mhlo.dynamic_reshape"(%tensor, %shape)  -> tensor<?x4xf32>
// %2 = tensor.cast(%t) tensor<?x?xf32> -> tensor<?x4xf32>
//
struct DynamicReshapeOpPartialShapeInference
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  using OpRewritePattern<mhlo::DynamicReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto output_shape =
        op.output_shape().getDefiningOp<tensor::FromElementsOp>();
    if (!output_shape) {
      return failure();
    }
    auto result_type = op.getResult().getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> result_dims(result_type.getRank());
    bool has_uninfered_static_dim = false;
    for (auto element : llvm::enumerate(output_shape.getElements())) {
      int64_t new_value = -1;
      if (result_type.isDynamicDim(element.index())) {
        if (arith::ConstantIntOp constant_op =
                element.value().getDefiningOp<arith::ConstantIntOp>()) {
          new_value = constant_op.getValue().cast<IntegerAttr>().getInt();
        } else if (arith::ConstantIndexOp constant_op =
                       element.value()
                           .getDefiningOp<arith::ConstantIndexOp>()) {
          new_value = constant_op.getValue().cast<IntegerAttr>().getInt();
        }
      }

      if (new_value != -1) {
        has_uninfered_static_dim = true;
        result_dims[element.index()] = new_value;
      } else {
        result_dims[element.index()] = result_type.getDimSize(element.index());
      }
    }
    if (!has_uninfered_static_dim) {
      return failure();
    }
    auto new_type = result_type.clone(result_dims);
    auto new_op = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, new_type, op.operand(), output_shape);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), new_op);
    return success();
  }
};

// Canonicalizes
// %cst_shape = constant dense<[a, b, .. c]> : tensor<nxi32>
// %0 = "mhlo.dynamic_reshape"(%tensor, %cst_shape)  -> tensor<?x?x?xf32>
//
// into:
//
// %cst_shape = constant dense<[a, b, .. c]> : tensor<nxi32>
// %t = "mhlo.dynamic_reshape"(%tensor, %cst_shape)  -> tensor<axbxcxf32>
// %1 = tensor.cast(%t) tensor<?x?x?xf32> -> tensor<axbxcxf32>
class DynamicReshapeOpShapeInference
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* shape_def_op = op.output_shape().getDefiningOp();
    if (!shape_def_op) return failure();
    DenseIntElementsAttr cst_attr;
    if (auto cst_shape = dyn_cast<arith::ConstantOp>(shape_def_op)) {
      cst_attr = cst_shape.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
    } else if (auto mhlo_cst_shape = dyn_cast<mhlo::ConstantOp>(shape_def_op)) {
      cst_attr =
          mhlo_cst_shape.value().dyn_cast_or_null<DenseIntElementsAttr>();
    }
    if (!cst_attr) return failure();
    auto elem_ty = cst_attr.getType().cast<ShapedType>().getElementType();
    SmallVector<int64_t, 4> dims;
    if (elem_ty.isInteger(64) || elem_ty.isIndex()) {
      std::copy(cst_attr.getValues<int64_t>().begin(),
                cst_attr.getValues<int64_t>().end(), std::back_inserter(dims));
    } else if (elem_ty.isInteger(32)) {
      std::copy(cst_attr.getValues<int32_t>().begin(),
                cst_attr.getValues<int32_t>().end(), std::back_inserter(dims));
    } else {
      return failure();
    }
    auto result_ty = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) return failure();
    RankedTensorType new_ty =
        RankedTensorType::get(dims, result_ty.getElementType());
    if (new_ty == result_ty) return failure();
    auto new_reshape = rewriter.create<mhlo::DynamicReshapeOp>(
        op.getLoc(), new_ty, op.operand(), op.output_shape());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), new_reshape);
    return success();
  }
};

// Convert an expand-like reshape op followed by a dynamic-broadcast-in-dim op
// to a single dynamic-broadcast-in-dim op. The dim size of expanded dims should
// be one.
//
// An example is like following:
//
//  %0 = ...: tensor<?x?xf32>
//  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//  %new_shape = tensor.from_elements %d0, %d1, %c1 : tensor<3xindex>
//  %1 = "mhlo.dynamic_reshape"(%0, %new_shape) : (tensor<?x?xf32>,
//  tensor<3xindex>) -> tensor<?x?x1xf32> %2 =
//  "mhlo.dynamic_broadcast_in_dim"(%1, %...) {broadcast_dimensions = dense<[0,
//  1, 2]> : tensor<3xi64>} : (tensor<?x?x1xf32>, tensor<3xindex>) ->
//  tensor<?x?x?xf32>
//
// This pattern will be converted to:
//
//  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %...) {broadcast_dimensions =
//  dense<[0, 1]> : tensor<3xi64>} : (tensor<?x?xf32>, tensor<3xindex>) ->
//  tensor<?x?x?xf32>
//
// Note that the reshape op can be either `mhlo::DynamicReshapeOp` or
// `mhlo::ReshapeOp`. The expanded dim can be in any axis.
class DynamicBroadcastInDimOpSimplifier
    : public OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto dynReshapeOp = dyn_cast_or_null<mhlo::DynamicReshapeOp>(
        op->getOperand(0).getDefiningOp());
    auto staticReshapeOp =
        dyn_cast_or_null<mhlo::ReshapeOp>(op->getOperand(0).getDefiningOp());
    if (!dynReshapeOp && !staticReshapeOp) {
      return failure();
    }

    auto bcastTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    Value reshapeResult = dynReshapeOp != nullptr ? dynReshapeOp.getResult()
                                                  : staticReshapeOp.getResult();
    auto reshapeTy = reshapeResult.getType().dyn_cast<RankedTensorType>();
    Value input = dynReshapeOp != nullptr ? dynReshapeOp.getOperand(0)
                                          : staticReshapeOp.getOperand();
    auto inputTy = input.getType().dyn_cast<RankedTensorType>();
    if (!bcastTy || !reshapeTy || !inputTy) {
      return failure();
    }

    if (bcastTy.getRank() != reshapeTy.getRank() ||
        inputTy.getRank() >= reshapeTy.getRank()) {
      return failure();
    }

    struct DimValue {
      int64_t value = -1;
      tensor::DimOp dynDimOp = nullptr;
    };
    SmallVector<DimValue> reshapeDimValues;
    if (staticReshapeOp != nullptr) {
      for (auto shape : reshapeTy.getShape()) {
        DimValue dimVal;
        dimVal.value = shape;
        reshapeDimValues.push_back(dimVal);
      }
    } else {
      auto fromElementsOp = dyn_cast_or_null<tensor::FromElementsOp>(
          dynReshapeOp->getOperand(1).getDefiningOp());
      if (!fromElementsOp) {
        return failure();
      }

      for (auto dim : fromElementsOp->getOperands()) {
        // Deal with static dim.
        int64_t staticVal = -1;
        if (arith::ConstantIntOp constant_op =
                dyn_cast_or_null<arith::ConstantIntOp>(dim.getDefiningOp())) {
          staticVal = constant_op.getValue().cast<IntegerAttr>().getInt();
          assert(staticVal > 0);
        } else if (arith::ConstantIndexOp constant_op =
                       dyn_cast_or_null<arith::ConstantIndexOp>(
                           dim.getDefiningOp())) {
          staticVal = constant_op.getValue().cast<IntegerAttr>().getInt();
          assert(staticVal > 0);
        }
        if (staticVal != -1) {
          DimValue dimVal;
          dimVal.value = staticVal;
          reshapeDimValues.push_back(dimVal);
          continue;
        }

        // Deal with dynamic dim.
        Value dynVal = dim;
        if (auto indexCastOp =
                dyn_cast_or_null<arith::IndexCastOp>(dynVal.getDefiningOp())) {
          dynVal = indexCastOp->getOperand(0);
        }
        auto dimOp = dyn_cast_or_null<tensor::DimOp>(dynVal.getDefiningOp());
        if (!dimOp || dimOp.getSource() != input) {
          return failure();
        }
        DimValue dimVal;
        dimVal.dynDimOp = dimOp;
        reshapeDimValues.push_back(dimVal);
      }
    }

    // Map between input dims and reshape dims. Every input dim should be mapped
    // to one reshape dim in order. The mapped dims are bcastDims. Other dims
    // should be constant one.
    SmallVector<int64_t> bcastDims;
    int64_t reshapeDimIdx = 0;
    auto inputShape = inputTy.getShape();
    for (std::size_t i = 0; i < inputTy.getRank(); i++) {
      bool matched = false;
      for (; reshapeDimIdx < reshapeTy.getRank() && !matched; reshapeDimIdx++) {
        // Either both kDynamicSize, or the same static-shape value.
        if (reshapeDimValues[reshapeDimIdx].value == inputShape[i]) {
          // Check dynamic dim value.
          if (inputShape[i] == ShapedType::kDynamicSize) {
            // It should be the dim-op of input's i-th dim. That is, the `index`
            // of dim-op should be `i`. Note that we already checked that the
            // source of dim-op is the input of reshape.
            auto dimOp = reshapeDimValues[reshapeDimIdx].dynDimOp;
            auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(
                dimOp.getIndex().getDefiningOp());
            if (!indexOp ||
                indexOp.getValue().cast<IntegerAttr>().getInt() != i) {
              return failure();
            }
          }
          bcastDims.push_back(reshapeDimIdx);
          matched = true;
        }
        // Non matched dim on the path should be constant one.
        if (!matched && reshapeDimValues[reshapeDimIdx].value != 1) {
          return failure();
        }
      }
      if (!matched) {
        // Failed to match.
        return failure();
      }
    }

    RankedTensorType ty = RankedTensorType::get(
        {static_cast<int64_t>(bcastDims.size())}, rewriter.getIntegerType(64));
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, op.getType(), input, op->getOperand(1),
        DenseIntElementsAttr::get(ty, bcastDims));

    return success();
  }
};

struct ShapeSimplifierPass
    : public DiscShapeSimplifierPassBase<ShapeSimplifierPass> {
  ShapeSimplifierPass(const std::string& entry_func_name, bool insert_tie_shape)
      : DiscShapeSimplifierPassBase<
            ShapeSimplifierPass>::DiscShapeSimplifierPassBase() {
    this->entry_func_name_ = entry_func_name;
    this->insert_tie_shape_ = insert_tie_shape;
  }

  // Adds canonicalization patterns to the list of patterns.
  void AddCanonicalizationPatterns(MLIRContext* context,
                                   RewritePatternSet* patterns) {
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(*patterns, context);
  }

  void populateShapeRefinerPatterns(RewritePatternSet&);

  void runOnOperation() override;

  LogicalResult applyShapeAnalysis(ShapeAnalysisDeprecated&, bool&);

  LogicalResult applySymbolicShapeOptimization(ShapeAnalysisDeprecated&, bool&);
};

void ShapeSimplifierPass::populateShapeRefinerPatterns(
    RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
      ExtractFromExtentTensorCanonicalizationPattern,
      DynamicBroadcastInDimOpSimplifier,
      // TODO: upstream these general purpose patterns to hlo_ops.cc
      DynamicReshapeOpPartialShapeInference,
      DynamicReshapeOpShapeInference
  >(patterns.getContext());
  // clang-format on

  // Adds canonicalization patterns to the list of patterns.
  AddCanonicalizationPatterns(patterns.getContext(), &patterns);
}

void ShapeSimplifierPass::runOnOperation() {
  ModuleOp m = getOperation();
  func::FuncOp main = m.lookupSymbol<func::FuncOp>(entry_func_name_);
  if (!main) {
    m.emitError("entry func: " + entry_func_name_ + " not found");
    signalPassFailure();
    return;
  }

  bool changed = true;
  while (changed) {
    // suppose not change the IR by default.
    changed = false;

    // Stage #1: refine shape information locally
    // - Initialize local shape refiner patterns.
    RewritePatternSet patterns(main.getContext());
    populateShapeRefinerPatterns(patterns);
    // - apply these patterns
    // ignore the not-converged error since we are in a loop.
    (void)applyPatternsAndFoldGreedily(main, std::move(patterns));

    // Stage #2: propagate shape information globally
    ShapeAnalysisDeprecated analysis(main);
    if (failed(analysis.run())) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    if (failed(applyShapeAnalysis(analysis, changed))) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    // Stage #3: apply symbolic shape optimization. e.g. %out = bcast(%in) ->
    // %out = %in
    if (failed(applySymbolicShapeOptimization(analysis, changed))) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }

    // In the last interation, we explicitly insert tie_shape op to connect
    // symbol equal dimensions in the IR level.
    if (!changed && insert_tie_shape_ && failed(analysis.buildTieShapeOps())) {
      // error message should be generated inside the above function call.
      signalPassFailure();
      return;
    }
  }
}

LogicalResult ShapeSimplifierPass::applyShapeAnalysis(
    ShapeAnalysisDeprecated& analysis, bool& changed) {
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(analysis.getOperation());
  if (func == nullptr) {
    return failure();
  }
  auto updateIfNotSame = [&](Value value) {
    Type refinedTy = analysis.getRefinedType(value);
    if (refinedTy != value.getType()) {
      changed = true;
      value.setType(refinedTy);
    }
  };

  // apply refined type for each value
  func.walk([&](Operation* op) {
    for (Value operand : op->getOperands()) updateIfNotSame(operand);

    for (Value result : op->getResults()) updateIfNotSame(result);
  });

  // apply refined function type
  // 1, collect input types
  SmallVector<Type, 4> refinedInputTypes;
  for (Value arg : func.getArguments())
    refinedInputTypes.push_back(analysis.getRefinedType(arg));

  // 2, collect output types
  SmallVector<Type, 4> refinedOutputTypes;
  assert(func.getBody().getBlocks().size() == 1);
  Operation& op = func.getBody().front().getOperations().back();
  for (Value operand : op.getOperands())
    refinedOutputTypes.push_back(analysis.getRefinedType(operand));

  // 3, refine function type to new type
  auto newFuncTy = FunctionType::get(func.getContext(), refinedInputTypes,
                                     refinedOutputTypes);
  if (func.getFunctionType() != newFuncTy) {
    func.setType(newFuncTy);
    changed = true;
  }

  return success();
}

LogicalResult ShapeSimplifierPass::applySymbolicShapeOptimization(
    ShapeAnalysisDeprecated& analysis, bool& changed) {
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(analysis.getOperation());
  if (func == nullptr) {
    return failure();
  }

  SmallVector<Operation*, 4> mhloBcastOps;
  SmallVector<Operation*, 4> shapeBcastOps;
  func.walk([&](Operation* op) {
    if (isa<mhlo::DynamicBroadcastInDimOp>(op) &&
        analysis.isShapeEqual(op->getResult(0), op->getOperand(0))) {
      // %out = mhlo::bcast(%in, ...) -> shape_of(%in) == shape_of(%out)
      mhloBcastOps.push_back(op);
    } else if (isa<shape::BroadcastOp>(op) &&
               analysis.isShapeValueEqual(op->getOperand(0),
                                          op->getOperand(1))) {
      // %out = shape::bcast(%shape0, %shape1) -> shape0 is equal with shape1
      shapeBcastOps.push_back(op);
    }
  });
  for (Operation* op : mhloBcastOps) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    changed = true;
  }
  for (Operation* op : shapeBcastOps) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    changed = true;
  }

  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeSimplifierPass(
    const std::string& entry_func_name, bool insert_tie_shape) {
  return std::make_unique<ShapeSimplifierPass>(entry_func_name,
                                               insert_tie_shape);
}

}  // namespace disc_ral
}  // namespace mlir
