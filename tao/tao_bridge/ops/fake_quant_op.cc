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
    OP_REQUIRES_OK(context, context->GetAttr("use_signed", &use_signed_));
    OP_REQUIRES_OK(context, context->GetAttr("use_symmetric", &use_symmetric_));
    OP_REQUIRES_OK(context, context->GetAttr("use_dynamic", &use_dynamic_));
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
  bool use_signed_;
  bool use_symmetric_;
  bool use_dynamic_;
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
    .Attr("use_signed: bool")
    .Attr("use_symmetric: bool")
    .Attr("use_dynamic: bool")
    .Attr("Tfloat: {float}")
    .Attr("Tint: {int32}")
    .SetIsStateful()
    .Doc("FakeQuant op to carry quant information. Implemented as Identity.")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_CPU),
                        DiscFakeQuantOp);
REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_GPU),
                        DiscFakeQuantOp);

}  // namespace tensorflow
