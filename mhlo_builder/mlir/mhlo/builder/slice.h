#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

// The direct usage of the function is not recommended, because
// start_index & end_index must be normalized before it is called.
// It's recommended to use BuildDynamicSlice.
mlir::Value BuildDynamicSliceInternal(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& input,
                                      const mlir::Value& start_index,
                                      const mlir::Value& end_index,
                                      const mlir::Value& step,
                                      mlir_dim_t dim_index);

mlir::Value BuildDynamicSlice(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const mlir::Value& start_index,
                              const mlir::Value& end_index,
                              const mlir::Value& step, mlir_dim_t dim_index);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const SmallValueVec4& values);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& scalar);

std::tuple<mlir::Value, mlir::Value> BuildHalfSplit(mlir::OpBuilder& builder,
                                                    const mlir::Location& loc,
                                                    const mlir::Value& input,
                                                    mlir_dim_t dim_index);

mlir::Value BuildSelect(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& input,
                        const mlir::Value& select_index, mlir_dim_t dim_index);
}  // namespace mhlo
}  // namespace mlir