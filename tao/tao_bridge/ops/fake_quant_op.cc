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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// DiscFakeQuantOp, implemented as Identity, is used to carry fake quant
// information.
class DiscFakeQuantOp : public OpKernel {
 public:
  explicit DiscFakeQuantOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("quant_min", &quant_min_));
    OP_REQUIRES_OK(context, context->GetAttr("quant_max", &quant_max_));
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("signed", &signed_));
    OP_REQUIRES_OK(context, context->GetAttr("symmetric", &symmetric_));
    OP_REQUIRES_OK(context, context->GetAttr("dynamic", &dynamic_));
    OP_REQUIRES_OK(context, context->GetAttr("per_channel", &per_channel_));

    if (per_channel_) {
      OP_REQUIRES(context, axis_.size() > 0,
                  errors::InvalidArgument(
                      "Per-channel quantization requires non-empty axis."));
    } else {
      OP_REQUIRES(context, axis_.size() == 0,
                  errors::InvalidArgument(
                      "Per-tensor quantization requires empty axis"));
    }
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

 private:
  int64 quant_min_;
  int64 quant_max_;
  int64 num_bits_;
  std::vector<int64> axis_;
  bool signed_;
  bool symmetric_;
  bool dynamic_;
  bool per_channel_;
};

REGISTER_OP("DiscFakeQuant")
    .Input("input: Tfloat")
    .Input("scale: Tfloat")
    .Input("zero_point: Tint")
    .Output("output: Tfloat")
    .Attr("quant_min: int")
    .Attr("quant_max: int")
    .Attr("num_bits: int")
    .Attr("axis: list(int)")
    .Attr("signed: bool")
    .Attr("symmetric: bool")
    .Attr("dynamic: bool")
    .Attr("per_channel: bool")
    .Attr("Tfloat: {float}")
    .Attr("Tint: {int64}")
    .Doc("FakeQuant op to carry quant information. Implemented as Identity.")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_CPU),
                        DiscFakeQuantOp);
REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_GPU),
                        DiscFakeQuantOp);

}  // namespace tensorflow
