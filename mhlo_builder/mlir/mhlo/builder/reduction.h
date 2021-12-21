#pragma once
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"
#include "mlir/mhlo/builder/mlir_utils.h"

namespace mlir {
namespace mhlo {

// math: f(input, numel) = input / numel
mlir::Value BuildNumelReciprocal(mlir::OpBuilder& builder,
                                 const mlir::Location& loc,
                                 const mlir::Value& input,
                                 const mlir::Value& numel);

template <typename MathOp>
mlir::Value BuildReductionInitValue(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type) {
  return BuildHloConstZeroForType(builder, loc, elem_type);
}

// Builds body for reduce op by using the using the template binary op as the
// reducer op.
template <typename MathOp>
void BuildReduceBody(mlir::OpBuilder& builder, mlir::Region& body,
                     const mlir::Type& element_type) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block* block = builder.createBlock(&body);

  // Block arguments are scalars of the given element type.
  mlir::Type type = mlir::RankedTensorType::get(/*shape=*/{}, element_type);
  block->addArguments({type, type});

  mlir::Location loc = body.getLoc();
  auto reducer =
      builder.create<MathOp>(loc, block->getArgument(0), block->getArgument(1));
  builder.create<mlir::mhlo::ReturnOp>(loc, reducer.getResult());
}

template <typename MathOp, bool IsMean = false>
mlir::Value BuildReduction(mlir::OpBuilder& builder, const mlir::Location& loc,
                           const mlir::Value& init_value,
                           const mlir::Value& input,
                           const SmallVec4<mlir_dim_t>& reduce_dims,
                           bool keepdim = false) {
  auto elem_type = GetMlirTensorElemType(input);
  // build a reduction with initial value
  auto reduce_dims_attr = BuildI64ElementsAttr(builder, reduce_dims);
  auto reduction = builder.create<mlir::mhlo::ReduceOp>(loc, input, init_value,
                                                        reduce_dims_attr);
  BuildReduceBody<MathOp>(builder, reduction.body(), elem_type);

  auto result = reduction.getResult(0);
  if (IsMean) {
    // if it's mean then divide the accumulate results with number of elements
    auto numel = BuildHloNumelOfTensor(builder, loc, input, reduce_dims);
    result = BuildNumelReciprocal(builder, loc, result, numel);
  }

  if (keepdim) {
    // if keepdim, then unsqueeze tensor shape to original rank
    result = BuildUnsqueezeTensorShape(builder, loc, result, reduce_dims);
  }

  return result;
}

}  // namespace mhlo
}  // namespace mlir
