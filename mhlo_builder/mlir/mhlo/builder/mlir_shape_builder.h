#pragma once

#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
mlir::Value BuildHloDimSizeOfTensor(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    mlir_dim_t dim_index);

// Build an mlir subgraph that get the tensor's shape
SmallValueVec4 BuildDimSizeListOfTensor(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const SmallVec4<mlir_dim_t>& dims = {});

mlir::Value BuildShapeOfTensor(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& tensor);

// Build an mlir subgraph that reshape the tensor
mlir::Value BuildDynamicReshapeTensor(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& tensor,
                                      const SmallValueVec4& new_shape_vals);

// Build an mlir subgraph that returns a new tensor with
// dims of size 1 inserted at the specified position.
mlir::Value BuildUnsqueezeTensorShape(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& tensor,
                                      const SmallVec4<mlir_dim_t>& unsqz_dims);

// Build an mlir subgraph that returns a new tensor with
// dims of size 1 removed from the specified position.
//
// NB: The squeezed dim_sizes are considered to be 1,
// otherwise the compilation behaviors are undefined.
mlir::Value BuildSqueezeTensorShape(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    const SmallVec4<mlir_dim_t>& sqz_dims);

std::tuple<mlir::Value, SmallValueVec4> BuildCollapseTensorShape(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallVec4<mlir_dim_t>& clap_dims);

mlir::Value BuildExpandTensorShapeWithDhloDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& expand_dims,
    mlir_dim_t expand_pos);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const SmallValueVec4& values);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& scalar);

mlir::Value BuildPermute(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& input,
                         const SmallVec4<mlir_dim_t>& trans_dim_vec);
}  // namespace mhlo
}  // namespace mlir
