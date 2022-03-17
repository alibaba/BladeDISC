// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* ==================================================
 * None Maximum Suppression Operation - GPU Version
 *
 * Author:
 *   Yifan Lu (evanlu.lyf@alibaba-inc.com)
 *
 ================================================== */

#ifndef __NONE_MAXIMUM_SUPPRESSION_OP_GPU_DEVICE__
#define __NONE_MAXIMUM_SUPPRESSION_OP_GPU_DEVICE__

#if GOOGLE_CUDA

#include "src/custom_ops/non_max_suppression/gpu_non_max_suppression.cu.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/tensor_format.h"

#define EIGEN_USE_GPU

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// -------------------------------------------------------------------------------------

namespace {

Status NMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks.
  ShapeHandle boxes;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
  ShapeHandle scores;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
  ShapeHandle max_output_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
  ShapeHandle iou_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
  ShapeHandle score_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
  // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
  DimensionHandle unused;
  // The boxes[0] and scores[0] are both num_boxes.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // The boxes[1] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

  c->set_output(0, c->Vector(c->UnknownDim()));
  return Status::OK();
}

}  // namespace

// -------------------------------------------------------------------------------------

REGISTER_OP("BladeNonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Output("selected_indices: int32")
    .Attr("iou_threshold: float = 0.5")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("BladeNonMaxSuppressionV2")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Attr("T: {float} = DT_FLOAT")
    .Attr("T_threshold: {float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      ShapeHandle iou_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("BladeNonMaxSuppressionV3")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Input("score_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Attr("T: {float} = DT_FLOAT")
    .Attr("T_threshold: {float} = DT_FLOAT")
    .SetShapeFn(NMSShapeFn);

REGISTER_OP("BladeNonMaxSuppressionV4")
    .Input("boxes: T")
    .Input("scores: T")
    .Input("max_output_size: int32")
    .Input("iou_threshold: T_threshold")
    .Input("score_threshold: T_threshold")
    .Output("selected_indices: int32")
    .Output("valid_outputs: int32")
    .Attr("T: {float} = DT_FLOAT")
    .Attr("T_threshold: {float} = DT_FLOAT")
    .Attr("pad_to_max_output_size: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(NMSShapeFn(c));

      bool pad_to_max;
      TF_RETURN_IF_ERROR(c->GetAttr("pad_to_max_output_size", &pad_to_max));
      if (pad_to_max) {
        // If padded, overwrite the shape of the output to be static.
        DimensionHandle output_dim;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &output_dim));
        c->set_output(0, c->MakeShape({output_dim}));
      }
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

// -------------------------------------------------------------------------------------

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

namespace gpu = perftools::gputools;
template <typename T>
inline gpu::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  gpu::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  gpu::DeviceMemory<T> typed(wrapped);
  return typed;
}

// -------------------------------------------------------------------------------------

template <typename Device>
class BladeNonMaxSuppressionOp : public OpKernel {
 public:
  explicit BladeNonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("iou_threshold", &iou_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));

    OP_REQUIRES(context, iou_threshold_ >= 0 && iou_threshold_ <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const GPUDevice d = context->eigen_device<GPUDevice>();

    // If there is nothing to compute, return.
    if (boxes.shape().num_elements() == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }

    // Use temp device memory for NMS kernel output buffer, in case NMS
    // output size is less than max_output_size.
    const int temp_output_size = max_output_size.scalar<int>()();
    Tensor temp_output_indices;
    TensorShape temp_output_shape({temp_output_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, temp_output_shape,
                                                   &temp_output_indices));

    // Execute NMS filter
    int output_size = nmsEngine(
        context, d.stream(), d, num_boxes,
        temp_output_size, /* max output size */
        iou_threshold_, boxes.flat<float>().data(), scores.flat<float>().data(),
        temp_output_indices.flat<int>().data(), false /* avoid filter score */);

    Tensor* output_indices = nullptr;
    TensorShape output_shape({output_size});
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_indices));

    auto src_ptr = AsDeviceMemory(temp_output_indices.flat<int>().data(),
                                  temp_output_size);
    auto dst_ptr =
        AsDeviceMemory(output_indices->flat<int>().data(), output_size);

    // Copy NMS output indexes into output tensor.
    bool copy_status = stream
                           ->ThenMemcpyD2D(&dst_ptr, src_ptr,
                                           output_size * sizeof(int) /* size */)
                           .ok();
    if (!copy_status) {
      context->SetStatus(
          errors::Internal("Failed to copy nms filtered indexes into output"));
    }
  }

 private:
  float iou_threshold_;
};

template <typename Device, typename T>
class BladeNonMaxSuppressionV2Op : public OpKernel {
 public:
  explicit BladeNonMaxSuppressionV2Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const GPUDevice d = context->eigen_device<GPUDevice>();

    // If there is nothing to compute, return.
    if (boxes.shape().num_elements() == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }

    // Use temp device memory for NMS kernel output buffer, in case NMS
    // output size is less than max_output_size.
    const int temp_output_size = max_output_size.scalar<int>()();
    Tensor temp_output_indices;
    TensorShape temp_output_shape({temp_output_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, temp_output_shape,
                                                   &temp_output_indices));

    // Execute NMS filter
    int output_size = nmsEngine(context, d.stream(), d, num_boxes,
                                temp_output_size, /* max output size */
                                iou_threshold_val, boxes.flat<float>().data(),
                                scores.flat<float>().data(),
                                temp_output_indices.flat<int>().data(),
                                false /* avoid filter score */);

    Tensor* output_indices = nullptr;
    TensorShape output_shape({output_size});
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_indices));

    auto src_ptr = AsDeviceMemory(temp_output_indices.flat<int>().data(),
                                  temp_output_size);
    auto dst_ptr =
        AsDeviceMemory(output_indices->flat<int>().data(), output_size);

    // Copy NMS output indexes into output tensor.
    bool copy_status = stream
                           ->ThenMemcpyD2D(&dst_ptr, src_ptr,
                                           output_size * sizeof(int) /* size */)
                           .ok();
    if (!copy_status) {
      context->SetStatus(
          errors::Internal("Failed to copy nms filtered indexes into output"));
    }
  }
};

template <typename Device, typename T>
class BladeNonMaxSuppressionV3Op : public OpKernel {
 public:
  explicit BladeNonMaxSuppressionV3Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const GPUDevice d = context->eigen_device<GPUDevice>();

    // If there is nothing to compute, return.
    if (boxes.shape().num_elements() == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }

    // Use temp device memory for NMS kernel output buffer, in case NMS
    // output size is less than max_output_size.
    const int temp_output_size = max_output_size.scalar<int>()();
    Tensor temp_output_indices;
    TensorShape temp_output_shape({temp_output_size});
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, temp_output_shape,
                                                   &temp_output_indices));

    // Execute NMS filter
    int output_size = nmsEngine(
        context, d.stream(), d, num_boxes,
        temp_output_size, /* max output size */
        iou_threshold_val, boxes.flat<float>().data(),
        scores.flat<float>().data(), temp_output_indices.flat<int>().data(),
        true /* do filter score */, score_threshold_val);

    Tensor* output_indices = nullptr;
    TensorShape output_shape({output_size});
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_indices));

    auto src_ptr = AsDeviceMemory(temp_output_indices.flat<int>().data(),
                                  temp_output_size);
    auto dst_ptr =
        AsDeviceMemory(output_indices->flat<int>().data(), output_size);

    // Copy NMS output indexes into output tensor.
    bool copy_status = stream
                           ->ThenMemcpyD2D(&dst_ptr, src_ptr,
                                           output_size * sizeof(int) /* size */)
                           .ok();
    if (!copy_status) {
      context->SetStatus(
          errors::Internal("Failed to copy nms filtered indexes into output"));
    }
  }
};

template <typename Device, typename T>
class BladeNonMaxSuppressionV4Op : public OpKernel {
 public:
  explicit BladeNonMaxSuppressionV4Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const GPUDevice d = context->eigen_device<GPUDevice>();

    // If there is nothing to compute, return.
    if (boxes.shape().num_elements() == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }

    int num_valid_outputs;
    Tensor* output_indices = nullptr;

    if (pad_to_max_output_size_) {
      const int output_size = max_output_size.scalar<int>()();
      TensorShape output_shape({output_size});
      OP_REQUIRES_OK(
          context, context->allocate_output(0, output_shape, &output_indices));

      // Execute NMS filter
      num_valid_outputs = nmsEngine(
          context, d.stream(), d, num_boxes, output_size, /* max output size*/
          iou_threshold_val, boxes.flat<float>().data(),
          scores.flat<float>().data(), output_indices->flat<int>().data(),
          true /* do filter score */, score_threshold_val);
    } else {  // avoid padding to maximum output size
      // Use temp device memory for NMS kernel output buffer, in case NMS
      // output size is less than max_output_size.
      const int temp_output_size = max_output_size.scalar<int>()();
      Tensor temp_output_indices;
      TensorShape temp_output_shape({temp_output_size});
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT32, temp_output_shape,
                                            &temp_output_indices));

      // Execute NMS filter
      int num_valid_outputs = nmsEngine(
          context, d.stream(), d, num_boxes,
          temp_output_size, /* max output size*/
          iou_threshold_val, boxes.flat<float>().data(),
          scores.flat<float>().data(), temp_output_indices.flat<int>().data(),
          true /* do filter score */, score_threshold_val);

      TensorShape output_shape({num_valid_outputs});
      OP_REQUIRES_OK(
          context, context->allocate_output(0, output_shape, &output_indices));

      auto src_ptr = AsDeviceMemory(temp_output_indices.flat<int>().data(),
                                    temp_output_size);
      auto dst_ptr =
          AsDeviceMemory(output_indices->flat<int>().data(), num_valid_outputs);

      // Copy NMS output indexes into output tensor.
      bool copy_status =
          stream
              ->ThenMemcpyD2D(&dst_ptr, src_ptr,
                              num_valid_outputs * sizeof(int) /* size */)
              .ok();
      if (!copy_status) {
        context->SetStatus(errors::Internal(
            "Failed to copy nms filtered indexes into output"));
      }
    }

    // Allocate scalar output tensor for number of indices computed.
    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, tensorflow::TensorShape{}, &num_outputs_t));
    num_outputs_t->scalar<int32>().setConstant(num_valid_outputs);
  }

 private:
  bool pad_to_max_output_size_;
};

// -------------------------------------------------------------------------------------

REGISTER_KERNEL_BUILDER(Name("BladeNonMaxSuppression")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size"),
                        BladeNonMaxSuppressionOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("BladeNonMaxSuppressionV2")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size")
                            .HostMemory("iou_threshold"),
                        BladeNonMaxSuppressionV2Op<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("BladeNonMaxSuppressionV3")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size")
                            .HostMemory("iou_threshold")
                            .HostMemory("score_threshold"),
                        BladeNonMaxSuppressionV3Op<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("BladeNonMaxSuppressionV4")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("max_output_size")
                            .HostMemory("iou_threshold")
                            .HostMemory("score_threshold")
                            .HostMemory("valid_outputs"),
                        BladeNonMaxSuppressionV4Op<GPUDevice, float>);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // __NONE_MAXIMUM_SUPPRESSION_OP_GPU_DEVICE__
