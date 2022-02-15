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

#include <mlir/mhlo/builder/mlir_attr_utils.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/mlir_utils.h>
#include <mlir/mhlo/builder/slice.h>
#include <mlir/mhlo/builder/standard.h>

#include "common_utils/logging.h"
#include "common_utils/utils.h"
#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

// NB: aten::views, aten::reshapes and aten::transpose may return
// tensors that share the same underlying memory buffer as the originals.
// Thus the inplace operations on the tensors returned by these operations,
// would have side-effects on their inputs
//
// However, in MLIR we would always return tensors with underlying
// memory buffers owned by themselves.
// So, please make sure the tensors aliasing has been analyzed and
// the tensors have pass the inplace-safty check.
bool ConvertAtenView(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_tensor = node.input(0);
  auto jit_size = node.input(1);
  auto ml_tensor = ctx.GetMlirValue(jit_tensor);
  if (ctx.list_map.find(jit_size) == ctx.list_map.end()) {
    return false;
  }
  auto ml_dim_sizes = ctx.GetMlirValueList(jit_size);

  auto& builder = *ctx.builder;
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  auto ml_rank = ml_dim_sizes.size();
  TORCH_CHECK(result_type.getRank() == ml_rank);
  if (ml_rank == 0) {
    // if reshape output is a scalar
    auto result =
        builder.create<mlir::mhlo::ReshapeOp>(loc, result_type, ml_tensor);
    ctx.value_map[node.output(0)] = result.getResult();
  } else {
    // if reshape output is a tensor
    ml_dim_sizes = BuildStdScalarToHloDimType(builder, loc, ml_dim_sizes);
    auto new_shape =
        BuildResolveUnknownDimSizeI32(builder, loc, ml_tensor, ml_dim_sizes);
    auto result = builder.create<mlir::mhlo::DynamicReshapeOp>(
        loc, result_type, ml_tensor, new_shape);
    ctx.value_map[node.output(0)] = result.getResult();
  }
  return true;
}

bool ConvertAtenTranspose(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_tensor = node.input(0);
  auto jit_dim0 = node.input(1);
  auto jit_dim1 = node.input(2);
  bool is_const_dims = IsPrimConstant(*jit_dim0) && IsPrimConstant(*jit_dim1);
  if (!is_const_dims) {
    LOG(WARNING) << "Transpose dimensions must be constant";
    return false;
  }

  auto ml_tensor = ctx.GetMlirValue(jit_tensor);
  mlir_dim_t rank = GetRankOfMlirValue(ml_tensor);
  mlir_dim_t trans_dim0 = CastJitConstToInt64(*jit_dim0);
  mlir_dim_t trans_dim1 = CastJitConstToInt64(*jit_dim1);
  trans_dim0 = NormalizeDimIndex(trans_dim0, rank);
  trans_dim1 = NormalizeDimIndex(trans_dim1, rank);

  SmallVec4<mlir_dim_t> trans_dim_vec = RangeIndices(0, rank);
  std::swap(trans_dim_vec[trans_dim0], trans_dim_vec[trans_dim1]);
  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] =
      BuildPermute(builder, loc, ml_tensor, trans_dim_vec);
  return true;
}

bool ConvertAtenT(MhloConversionContext& ctx, const torch::jit::Node& node) {
  // Ref: https://pytorch.org/docs/stable/generated/torch.t.html
  auto ml_tensor = ctx.GetMlirValue(node.input(0));
  mlir_dim_t rank = GetRankOfMlirValue(ml_tensor);
  if (rank < 2) {
    // 0-D and 1-D tensors are returned as is.
    ctx.value_map[node.output(0)] = ml_tensor;
  } else if (rank == 2) {
    // When input is a 2-D tensor this is equivalent to transpose(input, 0, 1).
    auto& builder = *ctx.builder;
    const auto& loc = GetNodeLocation(ctx, node);
    ctx.value_map[node.output(0)] =
        BuildPermute(builder, loc, ml_tensor, {1, 0});
  } else {
    // It's illegal, ref: https://pytorch.org/docs/stable/generated/torch.t.html
    DLOG(ERROR)
        << "Illegal torch.t usage found, please reference to https://pytorch.org/docs/stable/generated/torch.t.html";
    return false;
  }

  return true;
}

bool ConvertAtenPermute(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto jit_dims = node.input(1);
  if (!CheckConstAttribute(jit_dims, "aten::permute", "dims")) {
    return false;
  }

  auto dims_ival = torch::jit::toIValue(jit_dims);
  if (!(dims_ival && dims_ival->isIntList())) {
    DLOG(WARNING) << "Permute dimensions must be constants";
    return false;
  }
  auto dims = dims_ival->toIntList();
  auto ml_tensor = ctx.GetMlirValue(node.input(0));
  mlir_dim_t rank = GetRankOfMlirValue(ml_tensor);
  TORCH_CHECK(rank == dims.size());
  SmallVec4<mlir_dim_t> dims_vec(dims.begin(), dims.end());
  SmallVec4<mlir_dim_t> trans_dim_vec = NormalizeDimIndex(dims_vec, rank);

  if (rank < 2) {
    ctx.value_map[node.output(0)] = ml_tensor;
    return true;
  }
  auto& builder = *ctx.builder;
  const auto& loc = GetNodeLocation(ctx, node);
  ctx.value_map[node.output(0)] =
      BuildPermute(builder, loc, ml_tensor, trans_dim_vec);
  return true;
}

bool ConvertAtenUnsqueeze(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto ml_tensor = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);

  if (!CheckConstAttribute(jit_dim, "aten::unsqueeze", "dim")) {
    return false;
  }

  mlir_dim_t dim = CastJitConstToInt64(*jit_dim);
  auto unsqz_tensor =
      BuildUnsqueezeTensorShape(*ctx.builder, loc, ml_tensor, {dim});
  ctx.value_map[node.output(0)] = unsqz_tensor;
  return true;
}

bool ConvertAtenSqueeze(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto jit_tensor = node.input(0);
  auto ml_tensor = ctx.GetMlirValue(jit_tensor);

  if (!GetTrustTracingShape()) {
    // Squeeze would leads to dynamic rank, and our shape are recorded by
    // tracing. We won't do conversion if we don't trust the tracing shape.
    //
    // TODO(gty):
    // This could be removed once we have more trustworthy shapes,
    // such as those analyzed from static shape analysis.
    return false;
  }

  auto tensor_type = jit_tensor->type()->cast<torch::TensorType>();
  TORCH_CHECK(tensor_type != nullptr);
  c10::optional<uint64_t> optional_rank = tensor_type->sizes().size();
  mlir_dim_t rank = 0;
  if (!optional_rank) {
    DLOG(WARNING) << "The tensor rank is unknown";
    return false;
  } else {
    rank = *optional_rank;
  }

  // NB: we would analyse rank information from the Tensor's shape,
  // so do shape analysis/inference and set the shape.
  // Otherwise, it's rank would be regarded as dynamic conservatively.
  auto tensor_sizes = *(tensor_type->sizes().concrete_sizes());
  SmallVec4<mlir_dim_t> sqz_dims;
  if (node.inputs().size() == 2) {
    auto jit_dim = node.input(1);

    if (!CheckConstAttribute(jit_dim, "aten::squeeze", "dim")) {
      return false;
    }

    mlir_dim_t dim = CastJitConstToInt64(*jit_dim);
    dim = NormalizeDimIndex(dim, rank);
    if (tensor_sizes[dim] == 1) {
      sqz_dims.push_back(dim);
    }
  } else {
    for (int k = 0; k < rank; ++k) {
      if (tensor_sizes[k] == 1) {
        sqz_dims.push_back(k);
      }
    }
  }

  if (sqz_dims.size() > 0) {
    const auto& loc = GetNodeLocation(ctx, node);
    auto sqz_tensor =
        BuildSqueezeTensorShape(*ctx.builder, loc, ml_tensor, sqz_dims);
    ctx.value_map[node.output(0)] = sqz_tensor;
  } else {
    ctx.value_map[node.output(0)] = ml_tensor;
  }
  return true;
}

bool ConvertAtenSlice(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto ml_tensor = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);
  auto start_val = ctx.GetMlirValue(node.input(2));
  auto end_val = ctx.GetMlirValue(node.input(3));
  auto step_val = ctx.GetMlirValue(node.input(4));
  if (!CheckConstAttribute(jit_dim, "aten::slice", "dim")) {
    return false;
  }
  auto& builder = *ctx.builder;
  auto dim_index = CastJitConstToInt64(*jit_dim);
  ctx.value_map[node.output(0)] = BuildDynamicSlice(
      builder, loc, ml_tensor, start_val, end_val, step_val, dim_index);
  return true;
}

bool ConvertAtenSelect(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto ml_tensor = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);
  auto ml_index = ctx.GetMlirValue(node.input(2));
  if (!CheckConstAttribute(jit_dim, "aten::select", "dim")) {
    return false;
  }
  auto& builder = *ctx.builder;
  auto dim_index = CastJitConstToInt64(*jit_dim);
  ctx.value_map[node.output()] =
      BuildSelect(builder, loc, ml_tensor, ml_index, dim_index);
  return true;
}

bool ConvertAtenSize(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto ml_tensor = ctx.GetMlirValue(node.input(0));

  auto& builder = *ctx.builder;
  if (node.inputs().size() > 1) {
    auto jit_dim = node.input(1);
    if (!CheckConstAttribute(jit_dim, "aten::size", "dim")) {
      return false;
    }
    auto dim_index = CastJitConstToInt64(*jit_dim);
    auto dim_size = BuildStdDimSizeOfTensor(builder, loc, ml_tensor, dim_index);
    ctx.value_map[node.output(0)] = dim_size;
  } else {
    ctx.list_map[node.output(0)] =
        BuildStdDimSizeListOfTensor(builder, loc, ml_tensor);
  }
  return true;
}

bool ConvertAtenRoll(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto jit_self = node.input(0);
  auto jit_shifts = node.input(1);
  auto jit_dims = node.input(2);
  if (!CheckConstAttribute(jit_shifts, "aten::roll", "shifts")) {
    return false;
  }
  if (!CheckConstAttribute(jit_dims, "aten::roll", "dims")) {
    return false;
  }

  auto shifts_ival = torch::jit::toIValue(jit_shifts);
  if (!(shifts_ival && shifts_ival->isIntList())) {
    DLOG(WARNING) << "aten::roll shifts must be constants";
    return false;
  }

  auto dims_ival = torch::jit::toIValue(jit_dims);
  if (!(dims_ival && dims_ival->isIntList())) {
    DLOG(WARNING) << "aten::roll dims must be constants";
    return false;
  }
  auto shifts = shifts_ival->toIntList();
  auto dims = dims_ival->toIntList();

  mlir::Value self = ctx.GetMlirValue(jit_self);
  auto& builder = *ctx.builder;
  for (size_t d = 0; d < dims.size(); ++d) {
    self = mlir::mhlo::BuildRoll(builder, loc, self, shifts[d], dims[d]);
  }
  ctx.value_map[node.output()] = self;

  return true;
}
bool ConvertAtenUnbind(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  if (!GetTrustTracingShape()) {
    // Doesn't support dynamic outputs, and our shape are recorded by
    // tracing. We won't do conversion if we don't trust the tracing shape.
    //
    // TODO(gty):
    // This could be removed once we have more trustworthy shapes,
    // such as those analyzed from static shape analysis.
    return false;
  }

  const auto& loc = GetNodeLocation(ctx, node);
  auto jit_self = node.input(0);
  auto jit_dim = node.input(1);

  if (!CheckConstAttribute(jit_dim, "aten::unbind", "dim")) {
    return false;
  }
  mlir_dim_t dim = CastJitConstToInt64(*jit_dim);

  auto tensor_type = jit_self->type()->cast<torch::TensorType>();
  TORCH_CHECK(tensor_type != nullptr);

  // NB: we get number of outputs from the Tensor's shape.
  auto tensor_sizes = *(tensor_type->sizes().concrete_sizes());
  auto dim_size = tensor_sizes[dim];
  auto self = ctx.GetMlirValue(jit_self);
  auto& builder = *ctx.builder;
  SmallVec4<mlir::Value> outputs;
  for (int64_t d = 0; d < dim_size; ++d) {
    // select(self, d, dim)
    auto std_index = BuildStdConstForI64(builder, loc, d);
    auto out = BuildSelect(builder, loc, self, std_index, dim);
    outputs.push_back(out);
  }
  ctx.list_map[node.output()] = outputs;
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
            ConvertAtenView)
        .pattern(
            // aten::reshape may return a view of tensor:
            // https://github.com/pytorch/pytorch/blob/69f6d94/aten/src/ATen/native/TensorShape.cpp#L855-L858.
            // However, the mhlo converter would build a reshape op that return
            // a copy
            "aten::reshape(Tensor self, int[] shape) -> Tensor",
            ConvertAtenView)
        .pattern(
            "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
            ConvertAtenTranspose)
        .pattern(
            "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
            ConvertAtenPermute)
        .pattern("aten::t(Tensor(a) self) -> (Tensor(a))", ConvertAtenT)
        .pattern(
            "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
            ConvertAtenUnsqueeze)
        .pattern(
            "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
            ConvertAtenSqueeze)
        .pattern(
            "aten::squeeze(Tensor(a) self) -> Tensor(a)",
            ConvertAtenSqueeze)
        .pattern(
            "aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)",
            ConvertAtenSlice)
        .pattern("aten::size.int(Tensor self, int dim) -> int", ConvertAtenSize)
        .pattern("aten::size(Tensor self) -> int[]", ConvertAtenSize)
        .pattern(
            "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
            ConvertAtenSelect)
        .pattern(
            "aten::unbind.int(Tensor(a) self, int dim=0) -> (Tensor[])",
            ConvertAtenUnbind)
        .pattern(
            "aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)",
            ConvertAtenRoll);
} // namespace
} // namespace blade
} // namespace torch
