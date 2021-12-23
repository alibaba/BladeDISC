#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h> // from tf repo
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/mhlo/builder/element_wise_binary.h>
#include <mlir/mhlo/builder/matmul.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/mlir_utils.h>

#include "compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

bool ConvertAtenMatmul(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  // There two similar operation in xla:
  // Dot: https://www.tensorflow.org/xla/operation_semantics#dot
  // DotGeneral: https://www.tensorflow.org/xla/operation_semantics#dotgeneral
  //
  // We choose DotGenernal here because it support batch-matmul.
  // However, aten::matmul batch dimensions are broadcastable, but
  // xla::DotGeneral is not.
  //
  // The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  // broadcastable). For example: if input is a (j×1×n×m) tensor and other is a
  // (k×m×p) tensor, out will be an (j×k×n×p) tensor.

  // TODO(gty): to support broadcast
  auto loc = GetNodeLocation(ctx, node);
  auto inp0 = node.input(0);
  auto inp1 = node.input(1);
  auto lhs = ctx.GetMlirValue(inp0);
  auto rhs = ctx.GetMlirValue(inp1);
  mlir_dim_t rank0 = GetRankOfMlirValue(lhs);
  mlir_dim_t rank1 = GetRankOfMlirValue(rhs);
  if (rank0 < 1 || rank1 < 1) {
    return false;
  }

  auto& builder = *ctx.builder;
  // The implementation reference to:
  // https://pytorch.org/docs/stable/generated/torch.matmul.html
  if (rank1 == 1) {
    if (rank0 == 1) {
      // If both tensors are 1-dimensional, the dot product (scalar) is
      // returned.
      lhs = BuildUnsqueezeTensorShape(builder, loc, lhs, {0});
      auto output = BuildDotProduct_mv(builder, loc, lhs, rhs);
      output = BuildReshapeTensorToScalar(builder, loc, output);
      ctx.value_map[node.output(0)] = output;
    } else {
      // If the first argument is 2-dimensional and the second argument is
      // 1-dimensional, the matrix-vector product is returned.
      // NB: if rank0 > 2 reshape it to rank 2.
      auto output = BuildDotProduct_mv(builder, loc, lhs, rhs);
      ctx.value_map[node.output(0)] = output;
    }
  } else if (rank1 == 2) {
    if (rank0 == 1) {
      // If the first argument is 1-dimensional, a 1 is prepended to its
      // dimension for the purpose of the batched matrix multiply and removed
      // after.
      lhs = BuildUnsqueezeTensorShape(builder, loc, lhs, {0});
      auto output = BuildDotProduct_mm(builder, loc, lhs, rhs);
      SmallValueVec4 claps_dim_values;
      std::tie(ctx.value_map[node.output(0)], claps_dim_values) =
          BuildCollapseTensorShape(builder, loc, output, {-2, -1});
    } else {
      // If both arguments are 2-dimensional, the matrix-matrix product is
      // returned. NB: if rank0 > 2 reshape it to rank 2.
      ctx.value_map[node.output(0)] =
          BuildDotProduct_mm(builder, loc, lhs, rhs);
    }
  } else {
    // rank1 > 2
    if (rank0 == 1) {
      // If the first argument is 1-dimensional, a 1 is prepended to its
      // dimension for the purpose of the batched matrix multiply and removed
      // after.
      lhs = BuildUnsqueezeTensorShape(builder, loc, lhs, {0});
      auto output = BuildDotProduct_bmm(builder, loc, lhs, rhs);
      SmallValueVec4 claps_dim_values;
      std::tie(ctx.value_map[node.output(0)], claps_dim_values) =
          BuildCollapseTensorShape(builder, loc, output, {-2, -1});
    } else {
      ctx.value_map[node.output(0)] =
          BuildDotProduct_bmm(builder, loc, lhs, rhs);
    }
  }
  return true;
}

bool ConvertAtenAddmm(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto input = ctx.GetMlirValue(node.input(0));
  auto mat1 = ctx.GetMlirValue(node.input(1));
  auto mat2 = ctx.GetMlirValue(node.input(2));
  auto beta = ctx.GetMlirValue(node.input(3));
  auto alpha = ctx.GetMlirValue(node.input(4));

  mlir_dim_t rank0 = GetRankOfMlirValue(mat1);
  mlir_dim_t rank1 = GetRankOfMlirValue(mat2);
  TORCH_CHECK(rank0 == 2, rank1 == 2);
  auto builder = *ctx.builder;
  auto prod = BuildDotProduct_mm(builder, loc, mat1, mat2);
  prod = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp, nullptr, true>(
      builder, loc, prod, alpha, GetMlirTensorElemType(prod));
  input = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp, nullptr, true>(
      builder, loc, input, beta, GetMlirTensorElemType(input));
  ctx.value_map[node.output()] = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, input, prod, GetMlirTensorElemType(input));
  return true;
}

// linear
bool ConvertAtenLinear(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto input = ctx.GetMlirValue(node.input(0));
  auto weight = ctx.GetMlirValue(node.input(1));
  ::llvm::Optional<mlir::Value> bias = ctx.GetOptionalMlirValue(node.input(2));

  // weight.T
  SmallVec4<mlir_dim_t> trans_dim_vec = {1, 0};
  auto& builder = *ctx.builder;
  auto transposed_weight = BuildPermute(builder, loc, weight, trans_dim_vec);

  // x*w^T
  auto out = BuildDotProduct_mm(builder, loc, input, transposed_weight);

  if (bias) {
    out = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
        builder, loc, out, *bias, GetMlirTensorElemType(out));
  }

  ctx.value_map[node.output()] = out;
  return true;
}

namespace {
// >>> # vector x vector
// >>> tensor1 = torch.randn(3)
// >>> tensor2 = torch.randn(3)
// >>> torch.matmul(tensor1, tensor2).size()
// torch.Size([])
//
// >>> # matrix x vector
// >>> tensor1 = torch.randn(3, 4)
// >>> tensor2 = torch.randn(4)
// >>> torch.matmul(tensor1, tensor2).size()
// torch.Size([3])
//
// >>> # batched matrix x broadcasted vector
// >>> tensor1 = torch.randn(10, 3, 4)
// >>> tensor2 = torch.randn(4)
// >>> torch.matmul(tensor1, tensor2).size()
// torch.Size([10, 3])
//
// >>> # batched matrix x batched matrix
// >>> tensor1 = torch.randn(10, 3, 4)
// >>> tensor2 = torch.randn(10, 4, 5)
// >>> torch.matmul(tensor1, tensor2).size()
// torch.Size([10, 3, 5])
//
// >>> # batched matrix x broadcasted matrix
// >>> tensor1 = torch.randn(10, 3, 4)
// >>> tensor2 = torch.randn(4, 5)
// >>> torch.matmul(tensor1, tensor2).size()
// torch.Size([10, 3, 5])
//
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            // Ref: https://pytorch.org/docs/stable/generated/torch.matmul.html
            "aten::matmul(Tensor self, Tensor other) -> Tensor",
            ConvertAtenMatmul)
        .pattern(
            // Ref: https://pytorch.org/docs/stable/generated/torch.addmm.html
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar "
            "beta=1, Scalar alpha=1) -> Tensor",
            ConvertAtenAddmm)
        .pattern(
            "aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
            ConvertAtenLinear);
} // namespace
} // namespace blade
} // namespace torch
