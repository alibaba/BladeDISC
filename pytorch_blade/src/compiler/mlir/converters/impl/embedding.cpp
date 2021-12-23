#include <mlir/mhlo/builder/gather.h>

#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_conversion_context.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {

bool ConvertAtenEmbedding(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  // Ref: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
  // padding_idx (int, optional)
  //  – If given, pads the output with the embedding vector at padding_idx
  //  (initialized to zeros) whenever it encounters the index.
  // scale_grad_by_freq (boolean, optional)
  //  – If given, this will scale gradients by the inverse of frequency of the
  //  words in the mini-batch. Default False.
  // sparse (bool, optional)
  //  – If True, gradient w.r.t. weight matrix will be a sparse tensor. See
  //  Notes for more details regarding sparse gradients.
  //
  if (!CheckConstAttribute(node.input(2), "aten::embedding", "padding_idx")) {
    return false;
  }
  if (!CheckConstAttribute(
          node.input(3), "aten::embedding", "scale_grad_by_freq")) {
    return false;
  }
  if (!CheckConstAttribute(node.input(4), "aten::embedding", "sparse")) {
    return false;
  }

  //  NB:
  //  Only first 2 parameters are valid at inference time
  auto loc = GetNodeLocation(ctx, node);
  auto ml_weight = ctx.GetMlirValue(node.input(0));
  auto ml_indices = ctx.GetMlirValue(node.input(1));

  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] =
      mlir::mhlo::BuildGather(builder, loc, ml_weight, ml_indices);
  return true;
}

namespace {
auto mhlo_conversion = MhloConversionPatternRegister().pattern(
    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> (Tensor)",
    ConvertAtenEmbedding);
}
} // namespace blade
} // namespace torch
