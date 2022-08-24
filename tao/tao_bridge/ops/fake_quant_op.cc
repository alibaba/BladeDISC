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

// DiscFakeQuantOp, implemented as Identity, is used to carry fake quant information.
class DiscFakeQuantOp : public OpKernel {
public:
  explicit DiscFakeQuantOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};


REGISTER_OP("DiscFakeQuant")
  .Input("input: float")
  .Input("scale: float")
  .Input("zero_point: int64")
  .Attr("quant_min: int64")
  .Attr("quant_max: int64")
  .Attr("num_bits: int64")
  .Attr("axis: list(int64)")
  .Attr("signed: bool")
  .Attr("symmetric: bool")
  .Attr("dynamic: bool")
  .Attr("per_channel: bool")
  .Doc("FakeQuant op to carry quant information. Implemented as Identity.")
  .SetShapeFn(shape_inference::UnchangedShape);


REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_CPU), DiscFakeQuantOp);
REGISTER_KERNEL_BUILDER(Name("DiscFakeQuant").Device(DEVICE_GPU), DiscFakeQuantOp);

} // namespace tensorflow
