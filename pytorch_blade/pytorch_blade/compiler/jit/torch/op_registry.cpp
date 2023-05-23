/* From PyTorch:
- *
- * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
- * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
- * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
- * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
- * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
- * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
- * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon
- * Bottou, Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research
- * Institute (Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute
- * (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
- */

#include "pytorch_blade/compiler/jit/torch/op_registry.h"

// Location for Commonly Used Shape registries

namespace torch {
namespace blade {
using namespace torch::jit;
// Requirements:
//   dims           : preserved from the first argument
//   scalar type    : preserved from the first argument (doesn't have to
//                    match other arguments)
//   device         : always matching and preserved
//   tensor inputs  : *
//   tensor outputs : 1
// NB: those ops (with slight adjustments) are good candidates for restarts.
//     Knowing the type and device of weights or biases is usually enough to
//     infer the output type.
std::shared_ptr<SchemaSet> nn_ops_first_input_preserving() {
  // clang-format off
  std::shared_ptr<SchemaSet> ops = std::make_shared<SchemaSet>(
    SchemaSet {
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor",
      "aten::conv1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad) -> Tensor",
      "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
      "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
      "aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor",
      "aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor",
      "aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor",
      "aten::avg_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
      "aten::avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
      "aten::max_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor",
      "aten::max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor",
      "aten::reflection_pad1d(Tensor self, int[] padding) -> Tensor",
      "aten::reflection_pad2d(Tensor self, int[] padding) -> Tensor",
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 12
      "aten::reflection_pad3d(Tensor self, int[] padding) -> Tensor",
#endif
      "aten::replication_pad1d(Tensor self, int[] padding) -> Tensor",
      "aten::replication_pad2d(Tensor self, int[] padding) -> Tensor",
      "aten::replication_pad3d(Tensor self, int[] padding) -> Tensor",
      "aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners, float? scales) -> Tensor",
      "aten::upsample_nearest1d(Tensor self, int[] output_size, float? scales) -> Tensor",
      "aten::upsample_nearest2d(Tensor self, int[] output_size, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
      "aten::upsample_nearest3d(Tensor self, int[] output_size, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_trilinear3d(Tensor self, int[] output_size, bool align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
      "aten::prelu(Tensor self, Tensor weight) -> Tensor",

      // Added because Hardswish is really hard to convert to
      // metatensors
      "aten::hardswish(Tensor self) -> Tensor",
      "aten::hardswish_(Tensor self) -> Tensor",
  });
  // clang-format on
  return ops;
};

// Requirements:
//   dims           : Changed from first argument
//   scalar type    : preserved from the first argument
//   device         : always matching and preserved
//   tensor inputs  : 1
//   tensor outputs : 1
std::shared_ptr<SchemaSet> ops_one_tensor_in_shape_transform() {
  std::shared_ptr<SchemaSet> ops = std::make_shared<SchemaSet>(SchemaSet{
      "aten::flatten(Tensor self, int start_dim, int end_dim) -> Tensor",
  });
  return ops;
};
} // namespace blade
} // namespace torch
