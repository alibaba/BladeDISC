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

#include "mlir/mhlo/builder/convolution.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value
BuildConvolution(mlir::OpBuilder &builder, const mlir::Location &loc,
                 const mlir::Value &input, const mlir::Value &weight,
                 const mlir::Value &padding, mlir::ArrayRef<int64_t> strides,
                 mlir::ArrayRef<int64_t> dilations) {
  size_t n_spatial_dims = strides.size();
  MHLO_CHECK(n_spatial_dims == dilations.size(),
             "The number of spatial dims mismatch");

  auto stride_attr = builder.getNamedAttr(
      "window_strides", BuildI64ElementsAttr(builder, strides));
  auto dilation_attr = builder.getNamedAttr(
      "rhs_dilation", BuildI64ElementsAttr(builder, dilations));

  mlir::IntegerAttr batch_dim_attr = builder.getI64IntegerAttr(0);
  mlir::IntegerAttr feature_dim_attr = builder.getI64IntegerAttr(1);
  auto spatial_dims_attr =
      BuildI64ElementsAttr(builder, RangeIndices(2, 2 + n_spatial_dims));

  mlir::IntegerAttr kernel_input_feature_dim = builder.getI64IntegerAttr(1);
  mlir::IntegerAttr kernel_output_feature_dim = builder.getI64IntegerAttr(0);

  auto conv_dims_numbers = builder.getNamedAttr(
      "dimension_numbers",
      mlir::mhlo::ConvDimensionNumbers::get(
          batch_dim_attr, feature_dim_attr, spatial_dims_attr,
          kernel_input_feature_dim, kernel_output_feature_dim,
          spatial_dims_attr, batch_dim_attr, feature_dim_attr,
          spatial_dims_attr, builder.getContext()));

  auto batch_group_count_attr =
      builder.getNamedAttr("batch_group_count", builder.getI64IntegerAttr(1));
  auto feature_group_count_attr =
      builder.getNamedAttr("feature_group_count", builder.getI64IntegerAttr(1));

  SmallValueVec4 operands{input, weight, padding};
  auto output_type = BuildRankedTensorTypeFrom(input);
  mlir::NamedAttribute attrs[] = {dilation_attr, stride_attr, conv_dims_numbers,
                                  batch_group_count_attr,
                                  feature_group_count_attr};
  auto conv_op = builder.create<mlir::mhlo::DynamicConvOp>(
      loc, output_type, operands, llvm::makeArrayRef(attrs));
  return conv_op.getResult();
}

} // namespace mhlo
} // namespace mlir
