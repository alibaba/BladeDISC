/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file rewriters conv ops' padding value to match the format of CUDNN
// library call.
// cuDNN only supports padding the same amount on the left and right sides,
// and on the top and bottom sides. So we manually create a new padded
// input tensor such that we can pass it to cuDNN.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"    // TF:llvm-project
#include "mlir/IR/Attributes.h"               // TF:llvm-project
#include "mlir/IR/Builders.h"                 // TF:llvm-project
#include "mlir/IR/Location.h"                 // TF:llvm-project
#include "mlir/IR/MLIRContext.h"              // TF:llvm-project
#include "mlir/IR/Matchers.h"                 // TF:llvm-project
#include "mlir/IR/Operation.h"                // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:local_config_mlir
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "transforms/PassDetail.h"
#include "transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

struct DiscGpuConvPaddingLegalizationPass
    : public GpuConvPaddingLegalizationPassBase<
          DiscGpuConvPaddingLegalizationPass> {
  explicit DiscGpuConvPaddingLegalizationPass()
      : GpuConvPaddingLegalizationPassBase<DiscGpuConvPaddingLegalizationPass>::
            GpuConvPaddingLegalizationPassBase() {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  int rank;
  int num_spatial_dims;
  Value input;
  Value filter;
  Value padding;
  Value output;
  RankedTensorType input_tp;
  RankedTensorType filter_tp;
  RankedTensorType padding_tp;

  bool IsLegalConstantPadding(mhlo::DynamicConvOp op) {
    DenseIntElementsAttr dense_elem_attr;
    if (matchPattern(padding, m_Constant(&dense_elem_attr))) {
      auto int_values = dense_elem_attr.getValues<APInt>();
      auto it = int_values.begin();
      for (int i = 0; i < num_spatial_dims; ++i) {
        assert(it != int_values.end());
        int64_t padding_low = (*it++).getSExtValue();
        assert(it != int_values.end());
        int64_t padding_high = (*it++).getSExtValue();
        if (padding_low != padding_high) return false;
      }
      return true;
    }
    return false;
  }

  void InsertPaddingOp(mhlo::DynamicConvOp op) {
    Location loc = op.getLoc();
    OpBuilder b(op);
    auto shape_scalar_type = padding_tp.getElementType();
    Value zero = b.create<arith::ConstantIntOp>(loc, 0, shape_scalar_type);
    Value one = b.create<arith::ConstantIntOp>(loc, 1, shape_scalar_type);

    // We suppose this pass runs after the `dhlo_conv_rewriter` pass, thus we
    // have:
    //  - input format: NCHW
    //  - filter format: OIHW
    //  - output format: NCHW
    // TODO(disc): Re-visit this after we have a global layout optimization
    // pass.
    //
    // The first two dimensions are batch & filter and  don't need padding
    SmallVector<Value, 4> padding_low{zero, zero};
    SmallVector<Value, 4> padding_high{zero, zero};
    SmallVector<Value, 4> padding_interior(rank, zero);
    padding_low.reserve(rank);
    padding_high.reserve(rank);
    // Original:
    //   output = conv(input, filter, padding)
    // After rewrite:
    //   common_padding, other_padding = calculate_new_padding(...)
    //   padded_input = d_pad(input, other_padding, ...)
    //   output = conv(padded_input, filter, common_padding)
    // Where:
    // ```
    //  common_padding = []
    //  other_padding = []
    //  for (padding_low, padding_high) in padding:
    //    common_padding_for_this_dim = min(padding_low, padding_high)
    //    remaining_low_padding = padding_low - common_padding_for_this_dim
    //    remaining_high_padding = padding_high - common_padding_for_this_dim
    //    common_padding.append(common_padding_for_this_dim, ...)
    //    other_padding.append((remaining_low_padding, remaining_high_padding))
    // ```
    SmallVector<Value, 4> new_padding_for_conv;
    new_padding_for_conv.reserve(num_spatial_dims * 2);
    for (int i = 0; i < num_spatial_dims; ++i) {
      SmallVector<Value, 1> low_indices(
          1, b.create<arith::ConstantIndexOp>(loc, 2 * i));
      SmallVector<Value, 1> high_indices(
          1, b.create<arith::ConstantIndexOp>(loc, 2 * i + 1));
      Value low_value = b.create<tensor::ExtractOp>(loc, padding, low_indices);
      Value high_value =
          b.create<tensor::ExtractOp>(loc, padding, high_indices);
      Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                           low_value, high_value);
      Value common_value =
          b.create<mlir::SelectOp>(loc, pred, low_value, high_value);
      Value remaining_low_value =
          b.create<mlir::arith::SubIOp>(loc, low_value, common_value);
      Value remaining_high_value =
          b.create<mlir::arith::SubIOp>(loc, high_value, common_value);
      // same padding value for low & high after rewritering
      new_padding_for_conv.push_back(common_value);
      new_padding_for_conv.push_back(common_value);
      // remamining paddings are handled by an explicit padding op
      padding_low.push_back(remaining_low_value);
      padding_high.push_back(remaining_high_value);
    }
    Value padding_low_tensor =
        b.create<tensor::FromElementsOp>(loc, ArrayRef<Value>({padding_low}));
    Value padding_high_tensor =
        b.create<tensor::FromElementsOp>(loc, ArrayRef<Value>({padding_high}));
    Value padding_interior_tensor = b.create<tensor::FromElementsOp>(
        loc, ArrayRef<Value>({padding_interior}));
    Value new_padding_tensor_for_conv = b.create<tensor::FromElementsOp>(
        loc, ArrayRef<Value>({new_padding_for_conv}));

    SmallVector<int64_t, 4> padded_input_shape;
    padded_input_shape.push_back(input_tp.getShape()[0]);
    padded_input_shape.push_back(input_tp.getShape()[1]);
    for (int i = 0; i < num_spatial_dims; ++i) {
      padded_input_shape.push_back(ShapedType::kDynamicSize);
    }

    Value padding_value_tensor = b.create<mhlo::ConstOp>(
        loc, disc_ral::GetScalarOfType(input_tp.getElementType(), 0));
    auto padded_input_tp =
        RankedTensorType::get(padded_input_shape, input_tp.getElementType());
    Value padded_input = b.create<mhlo::DynamicPadOp>(
        loc, padded_input_tp, input, padding_value_tensor, padding_low_tensor,
        padding_high_tensor, padding_interior_tensor);
    op.getOperation()->setOperand(0, padded_input);
    op.getOperation()->setOperand(2, new_padding_tensor_for_conv);
  }

  void RewriteOp(mhlo::DynamicConvOp op) {
    input = op.lhs();
    filter = op.rhs();
    padding = op.d_padding();
    output = op.getResult();

    input_tp = input.getType().dyn_cast<RankedTensorType>();
    filter_tp = filter.getType().dyn_cast<RankedTensorType>();
    padding_tp = padding.getType().dyn_cast<RankedTensorType>();

    if (!input_tp || !filter_tp || !padding_tp) {
      op.emitOpError() << "operands must be ranked type";
      return;
    }

    Location loc = op.getLoc();
    rank = filter_tp.getRank();
    num_spatial_dims = rank - 2;

    if (num_spatial_dims < 1) {
      op.emitOpError() << "conv op's input rank is less than 3";
      return;
    }

    // We only support Conv2D ATM.
    if (num_spatial_dims != 2) {
      return;
    }

    // Returns directly if the padding is constant and even, nothing needs to be
    // done.
    if (IsLegalConstantPadding(op)) {
      return;
    }

    // In dynamic shape mode, we have to insert a padding op unconditionally
    // even though the padding is even actually since we can not infer this in
    // compile-time. This may lead to the performace slowdown. Another option is
    // to do the padding inside the conv kernel, which in turn loses the
    // opportunity to fuse the padding op. Currently, we choose to explicitly
    // insert a padding op. Re-visit this in case necessary.
    InsertPaddingOp(op);
  }

  void runOnFunction() override {
    SmallVector<mhlo::DynamicConvOp, 4> ops;
    getFunction().walk([&](mhlo::DynamicConvOp op) { ops.push_back(op); });

    for (auto& op : ops) {
      if (placement_utils::isGpuMhlo(op)) {
        RewriteOp(op);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscGpuConvPaddingLegalization() {
  return std::make_unique<DiscGpuConvPaddingLegalizationPass>();
}

}  // namespace disc_ral
}  // namespace mlir
