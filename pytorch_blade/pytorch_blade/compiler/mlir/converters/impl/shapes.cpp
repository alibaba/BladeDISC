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

#include <mlir/mhlo/builder/gather.h>
#include <mlir/mhlo/builder/mlir_attr_utils.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/mlir_utils.h>
#include <mlir/mhlo/builder/slice.h>
#include <mlir/mhlo/builder/standard.h>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

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
    // if reshape output is a rank-0 tensor
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
    LOG(ERROR)
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
    LOG(WARNING) << "Permute dimensions must be constants";
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

  auto tensor_type = jit_tensor->type()->cast<at::TensorType>();
  TORCH_CHECK(tensor_type != nullptr);
  c10::optional<uint64_t> optional_rank = tensor_type->sizes().size();
  mlir_dim_t rank = 0;
  if (!optional_rank) {
    LOG(WARNING) << "The tensor rank is unknown";
    return false;
  } else {
    rank = *optional_rank;
  }

  // NB: we would analyse rank information from the Tensor's shape,
  // so do shape analysis/inference and set the shape.
  // Otherwise, it's rank would be regarded as dynamic conservatively.
  auto tensor_sizes = tensor_type->sizes();
  SmallVec4<mlir_dim_t> sqz_dims;
  if (node.inputs().size() == 2) {
    auto jit_dim = node.input(1);

    if (!CheckConstAttribute(jit_dim, "aten::squeeze", "dim")) {
      return false;
    }

    mlir_dim_t dim = CastJitConstToInt64(*jit_dim);
    dim = NormalizeDimIndex(dim, rank);
    auto dsize = tensor_sizes[dim];
    if (dsize && *dsize == 1) {
      sqz_dims.push_back(dim);
    } else {
      LOG(WARNING) << "the squeeze dimension size is not 1";
    }
  } else {
    for (int k = 0; k < rank; ++k) {
      auto dsize = tensor_sizes[k];
      if (dsize && *dsize == 1) {
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
  auto jit_start = node.input(2);
  auto jit_end = node.input(3);
  if (!CheckConstAttribute(jit_dim, "aten::slice", "dim")) {
    return false;
  }

  auto start_ivalue = torch::jit::toIValue(jit_start);
  auto end_ivalue = torch::jit::toIValue(jit_end);
  auto start_isnone = start_ivalue && start_ivalue->isNone();
  auto end_isnone = end_ivalue && end_ivalue->isNone();
  if (start_isnone && end_isnone) {
    ctx.value_map[node.output(0)] = ml_tensor;
    return true;
  }

  auto& builder = *ctx.builder;
  auto dim_index = CastJitConstToInt64(*jit_dim);
  auto start_val = BuildStdConstForI64(builder, loc, 0);
  auto end_val = start_val;

  if (!start_isnone) {
    start_val = ctx.GetMlirValue(jit_start);
  }
  if (end_isnone) {
    end_val = BuildStdDimSizeOfTensor(builder, loc, ml_tensor, dim_index);
  } else {
    end_val = ctx.GetMlirValue(jit_end);
  }

  auto step_val = ctx.GetMlirValue(node.input(4));
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
    LOG(WARNING) << "aten::roll shifts must be constants";
    return false;
  }

  auto dims_ival = torch::jit::toIValue(jit_dims);
  if (!(dims_ival && dims_ival->isIntList())) {
    LOG(WARNING) << "aten::roll dims must be constants";
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

  auto tensor_type = jit_self->type()->cast<at::TensorType>();
  TORCH_CHECK(tensor_type != nullptr);

  // NB: we get number of outputs from the Tensor's shape.
  auto tensor_sizes = tensor_type->sizes();
  auto dim_size = tensor_sizes[dim];
  TORCH_CHECK(dim_size, "aten::unbind dimension can't be unknown");
  auto self = ctx.GetMlirValue(jit_self);
  auto& builder = *ctx.builder;
  SmallVec4<mlir::Value> outputs;
  for (int64_t d = 0; d < *dim_size; ++d) {
    // select(self, d, dim)
    auto std_index = BuildStdConstForI64(builder, loc, d);
    auto out = BuildSelect(builder, loc, self, std_index, dim);
    outputs.push_back(out);
  }
  ctx.list_map[node.output()] = outputs;
  return true;
}

bool ConvertAtenIndexSelect(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto input = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);
  auto dim = CastJitConstToInt64(*jit_dim);
  auto index = ctx.GetMlirValue(node.input(2));

  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] =
      mlir::mhlo::BuildGather(builder, loc, input, index, dim);
  return true;
}

bool ConvertAtenFlip(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto input = ctx.GetMlirValue(node.input(0));
  auto jit_dims = node.input(1);
  const char* op_name = node.kind().toDisplayString();
  if (!CheckConstAttribute(jit_dims, op_name, "dims")) {
    return false;
  }
  auto dims_ival = torch::jit::toIValue(jit_dims);
  if (!(dims_ival && dims_ival->isIntList())) {
    LOG(WARNING) << "Flip dimensions must be constants";
    return false;
  }
  auto dims = dims_ival->toIntList();
  mlir_dim_t rank = GetRankOfMlirValue(input);
  SmallVec4<mlir_dim_t> dims_vec(dims.begin(), dims.end());
  SmallVec4<mlir_dim_t> trans_dim_vec = NormalizeDimIndex(dims_vec, rank);

  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] = builder.create<mlir::mhlo::ReverseOp>(
      loc, input, BuildI64ElementsAttr(builder, trans_dim_vec));
  return true;
}

bool ConvertAtenViewAs(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_tensor = node.input(0);
  auto jit_other = node.input(1);
  auto ml_tensor = ctx.GetMlirValue(jit_tensor);
  auto ml_other = ctx.GetMlirValue(jit_other);

  auto& builder = *ctx.builder;
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));

  auto other_ty = ml_other.getType().cast<mlir::RankedTensorType>();
  auto other_shape = other_ty.getShape();
  auto rank = other_ty.getRank();
  ::llvm::SmallVector<mlir::Value> dim_values;
  for (int64_t idx = 0; idx < rank; ++idx) {
    mlir::Value dim_val = other_shape[idx] == mlir::ShapedType::kDynamicSize
        ? builder.create<mlir::tensor::DimOp>(loc, ml_other, idx).getResult()
        : builder.create<mlir::arith::ConstantIndexOp>(loc, other_shape[idx])
              .getResult();
    dim_values.push_back(dim_val);
  }
  mlir::Value new_shape =
      builder.create<mlir::tensor::FromElementsOp>(loc, dim_values);
  TORCH_CHECK(result_type.getRank() == rank);
  if (rank == 0) {
    // if reshape output is a rank-0 tensor
    auto result =
        builder.create<mlir::mhlo::ReshapeOp>(loc, result_type, ml_tensor);
    ctx.value_map[node.output(0)] = result.getResult();
  } else {
    // if reshape output is a tensor
    auto result = builder.create<mlir::mhlo::DynamicReshapeOp>(
        loc, result_type, ml_tensor, new_shape);
    ctx.value_map[node.output(0)] = result.getResult();
  }
  return true;
}

// chunk_size = (dim_size + chunks - 1) / chunks
// start_index = 0;
// for (int i = 0; i < chunks; ++i) {
//   end_index = start_index + chunk_size > dim_size ?
//               dim_size : start_index + chunk_size;
//   output[i] = mhlo.dynamic_slice(input, start_index, end_index, strides);
//   start_index = end_index;
// }
bool ConvertAtenChunk(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_input = node.input(0);
  auto jit_chunks = node.input(1);
  auto jit_dim = node.input(2);
  bool is_const_dims = IsPrimConstant(*jit_chunks) && IsPrimConstant(*jit_dim);
  if (!is_const_dims) {
    LOG(WARNING) << "chunks and dim must be constant for aten::chunk";
    return false;
  }

  auto& builder = *ctx.builder;
  auto ml_input = ctx.GetMlirValue(jit_input);
  mlir_dim_t rank = GetRankOfMlirValue(ml_input);
  mlir_dim_t trans_chunks = CastJitConstToInt64(*jit_chunks);
  mlir_dim_t trans_dim = CastJitConstToInt64(*jit_dim);
  trans_dim = NormalizeDimIndex(trans_dim, rank);
  auto dim_size = BuildStdDimSizeOfTensor(builder, loc, ml_input, trans_dim);
  auto chunks = BuildStdConstForI64(builder, loc, trans_chunks);
  auto one = BuildStdConstForI64(builder, loc, 1);
  auto dim_size_p_chunks_m_1 = builder.create<mlir::arith::SubIOp>(
      loc, builder.create<mlir::arith::AddIOp>(loc, dim_size, chunks), one);
  auto chunked_size =
      builder.create<mlir::arith::DivUIOp>(loc, dim_size_p_chunks_m_1, chunks);
  mlir::Value last_end_val = BuildStdConstForI64(builder, loc, 0);
  SmallVec4<mlir::Value> outputs;
  for (mlir_dim_t i = 0; i < trans_chunks; ++i) {
    mlir::Value end_val = BuildStdMinimumSigned(
        builder,
        loc,
        builder.create<mlir::arith::AddIOp>(loc, last_end_val, chunked_size),
        dim_size);
    outputs.push_back(BuildDynamicSliceInternal(
        builder, loc, ml_input, last_end_val, end_val, one, trans_dim));
    last_end_val = end_val;
  }
  ctx.list_map[node.output()] = outputs;
  return true;
}

bool ConvertAtenFlatten(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_tensor = node.input(0);
  auto jit_start_dim = node.input(1);
  auto jit_end_dim = node.input(2);

  if (!CheckConstAttribute(jit_start_dim, "aten::flatten", "start_dim")) {
    return false;
  }

  if (!CheckConstAttribute(jit_end_dim, "aten::flatten", "end_dim")) {
    return false;
  }

  auto ml_tensor = ctx.GetMlirValue(jit_tensor);
  mlir_dim_t rank = GetRankOfMlirValue(ml_tensor);
  mlir_dim_t start_dim = CastJitConstToInt64(*jit_start_dim);
  mlir_dim_t end_dim = CastJitConstToInt64(*jit_end_dim);

  auto& builder = *ctx.builder;
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  ::llvm::SmallVector<mlir::Value> dim_values;

  // In case the input is a rank-0 tensor, return a rank-1 flatten tensor
  // according to https://pytorch.org/docs/stable/generated/torch.flatten.html
  if (rank == 0) {
    dim_values.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, 1));
  } else {
    start_dim = (start_dim + rank) % rank;
    end_dim = (end_dim + rank) % rank;

    for (int64_t idx = 0; idx < rank; ++idx) {
      auto dz =
          builder.create<mlir::tensor::DimOp>(loc, ml_tensor, idx).getResult();
      if (idx > start_dim && idx < end_dim + 1) {
        dim_values[start_dim] =
            BuildStdMulSigned(builder, loc, dim_values[start_dim], dz);
      } else {
        dim_values.push_back(dz);
      }
    }
  }

  mlir::Value new_shape =
      builder.create<mlir::tensor::FromElementsOp>(loc, dim_values);
  TORCH_CHECK(result_type.getRank() == dim_values.size());
  // if reshape output is a tensor
  auto result = builder.create<mlir::mhlo::DynamicReshapeOp>(
      loc, result_type, ml_tensor, new_shape);
  ctx.value_map[node.output(0)] = result.getResult();
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
            "aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)",
            ConvertAtenViewAs)
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
            "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
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
            ConvertAtenRoll)
        .pattern(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
            ConvertAtenIndexSelect)
        .pattern(
            "aten::flip(Tensor self, int[] dims) -> Tensor",
            ConvertAtenFlip)
        .pattern(
            "aten::chunk(Tensor self, int chunks, int dim=0) -> (Tensor[])",
            ConvertAtenChunk)
        .pattern(
            "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)",
            ConvertAtenFlatten);
} // namespace

} // namespace blade
} // namespace torch
