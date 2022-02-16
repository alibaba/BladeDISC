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

#include "mlir/mhlo/builder/gather.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/standard.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildGather(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& params, const mlir::Value& indices,
                        mlir_dim_t axis) {
  auto params_rank_type = GetMilrRankedTensorType(params);
  auto indices_rank_type = GetMilrRankedTensorType(indices);

  auto mhlo_one = BuildStdConstForI32(builder, loc, 1);
  // slice_sizes
  bool need_d_gather = false;
  SmallValueVec4 slice_sizes;
  auto params_rank = params_rank_type.getRank();
  slice_sizes.reserve(params_rank);
  for (mlir_dim_t r = 0; r < params_rank; ++r) {
    if (r == axis) {
      slice_sizes.push_back(mhlo_one);
    } else {
      slice_sizes.push_back(BuildHloDimSizeOfTensor(builder, loc, params, r));
    }
  }
  auto slice_sizes_tensor = BuildFromElements(builder, loc, slice_sizes);

  // offset_dims
  SmallVec4<mlir_dim_t> offset_dims;
  auto indices_rank = indices_rank_type.getRank();
  for (mlir_dim_t r = 0; r < axis; ++r) {
    offset_dims.push_back(r);
  }
  for (mlir_dim_t r = axis + 1; r < params_rank; ++r) {
    offset_dims.push_back(r + indices_rank - 1);
  }

  // collapsed_slice_dims
  SmallVec4<mlir_dim_t> collapsed_slice_dims(1, axis);
  // start_index_map
  SmallVec4<mlir_dim_t> start_index_map(1, axis);
  // index_vector_dim
  mlir_dim_t index_vector_dim = indices_rank;
  auto dims_attr = GatherDimensionNumbersAttr::get(
      builder.getContext(),
      /*offset_dims=*/offset_dims,
      /*collapsed_slice_dims=*/collapsed_slice_dims,
      /*start_index_map=*/start_index_map,
      /*index_vector_dim=*/index_vector_dim);

  // output_shape = params.shape[:axis] + indices.shape +
  //                params.shape[axis + 1:]
  auto params_shape = params_rank_type.getShape();
  auto indices_shape = indices_rank_type.getShape();
  SmallVec4<mlir_dim_t> output_shape(params_shape.begin(),
                                     params_shape.begin() + axis);
  output_shape.insert(output_shape.end(), indices_shape.begin(),
                      indices_shape.end());
  output_shape.insert(output_shape.end(), params_shape.begin() + axis + 1,
                      params_shape.end());

  // create output tensor type
  auto output_type = mlir::RankedTensorType::get(
      output_shape, params_rank_type.getElementType());
  return builder
      .create<mlir::mhlo::DynamicGatherOp>(loc, output_type, params, indices,
                                           slice_sizes_tensor, dims_attr)
      .getResult();
}
}  // namespace mhlo
}  // namespace mlir
