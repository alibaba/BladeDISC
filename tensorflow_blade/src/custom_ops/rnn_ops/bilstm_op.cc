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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "cuda/include/cublas_v2.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

void BiLstmCellLauncher(cudaStream_t stream, const int hidden, const int batch,
                        const float* tmp_i, const float* tmp_i_bw,
                        const float* bias, float* h_out, float* h_out_bw,
                        const float* c_in, float* c_out, const float* wh,
                        const float* wh_bw, const float* h_in,
                        const float* h_in_bw);

void YConcatLauncher(cudaStream_t stream, const int hidden, const int batch,
                     const int length, const float* oy, const float* oy_bw,
                     float* oy_concat);

class BilstmOp : public OpKernel {
 public:
  cublasHandle_t handle;
  float fzero;
  float fone;
  explicit BilstmOp(OpKernelConstruction* context) : OpKernel(context) {
    cublasCreate(&handle);
    fzero = 0.0;
    fone = 1.0;
  }
  ~BilstmOp() { cublasDestroy(handle); }

  void Compute(OpKernelContext* context) override {
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* weight = nullptr;
    context->input("input", &input);
    context->input("input_h", &input_h);
    context->input("input_c", &input_c);
    context->input("weight", &weight);
    const int seq_length = input->dim_size(0);
    const int batch_size = input->dim_size(1);
    const int input_size = input->dim_size(2);
    const int num_units = input_h->dim_size(2);
    const int dir_count = 2;
    const int num_layers = input_h->dim_size(0) / dir_count;
    OP_REQUIRES(context,
                input_h->dim_size(0) % dir_count == 0 && num_layers >= 1,
                errors::InvalidArgument("Invalid input_h shape"));
    OP_REQUIRES(context, batch_size <= 32,
                errors::InvalidArgument(
                    "only batch_size <= 32 is supported for now while",
                    " while the current batch size is ", batch_size));
    Tensor* output = nullptr;
    auto output_shape =
        TensorShape({seq_length, batch_size, num_units * dir_count});
    context->allocate_output(0, output_shape, &output);
    Tensor* output_h = nullptr;
    auto output_h_shape = input_h->shape();
    context->allocate_output(1, output_h_shape, &output_h);
    Tensor* output_c = nullptr;
    auto output_c_shape = input_c->shape();
    context->allocate_output(2, output_c_shape, &output_c);

    // If there is nothing to compute, return.
    if (output_shape.num_elements() == 0 ||
        output_h_shape.num_elements() == 0 ||
        output_c_shape.num_elements() == 0) {
      return;
    }

    Tensor output_tmpi;
    context->allocate_temp(
        DT_FLOAT,
        TensorShape({dir_count, seq_length, batch_size, 4 * num_units}),
        &output_tmpi);
    Tensor output_wht;
    context->allocate_temp(DT_FLOAT,
                           TensorShape({dir_count, num_units, 4 * num_units}),
                           &output_wht);
    Tensor output_y;
    auto temp_output_shape =
        TensorShape({dir_count, seq_length, batch_size, num_units});
    context->allocate_temp(DT_FLOAT, temp_output_shape, &output_y);

    const float* ix = input->flat<float>().data();
    const float* ih = input_h->flat<float>().data();
    const float* ic = input_c->flat<float>().data();
    const float* wx = weight->flat<float>().data();
    const float* wh = wx + input_size * num_units * 4;
    const float* b = wx + dir_count * (input_size + num_units) * num_units * 4 +
                     (num_layers - 1) * dir_count *
                         (2 * num_units + num_units) * num_units * 4;
    float* oy_concat = output->flat<float>().data();
    float* oy = output_y.flat<float>().data();
    float* oh = output_h->flat<float>().data();
    float* oc = output_c->flat<float>().data();

    float* tmpi = output_tmpi.flat<float>().data();
    float* wht = output_wht.flat<float>().data();

    const float* ih_bw = ih + batch_size * num_units;
    const float* ic_bw = ic + batch_size * num_units;
    const float* wx_bw = wx + (input_size + num_units) * num_units * 4;
    const float* wh_bw = wh + (input_size + num_units) * num_units * 4;
    const float* b_bw = b + num_units * 8;

    float* wht_bw = wht + num_units * num_units * 4;
    float* oy_bw = oy + seq_length * batch_size * num_units;
    float* oh_bw = oh + batch_size * num_units;
    float* oc_bw = oc + batch_size * num_units;
    float* tmpi_bw = tmpi + seq_length * batch_size * num_units * 4;

    cublasSetStream(handle, d.stream());
    const int m = 4 * num_units;
    const int n = seq_length * batch_size;
    const int k = input_size;
    const int n1 = batch_size;
    const int k1 = num_units;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &fone, wx, k, ix, k,
                &fzero, tmpi, m);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &fone, wx_bw, k, ix,
                k, &fzero, tmpi_bw, m);

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k1, &fone, wh, k1, &fzero,
                wh, m, wht, m);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k1, &fone, wh_bw, k1,
                &fzero, wh_bw, m, wht_bw, m);

    int last = seq_length - 1;
    BiLstmCellLauncher(d.stream(), num_units, batch_size, tmpi,
                       tmpi_bw + last * m * n1, b, oy, oy_bw + last * k1 * n1,
                       ic, oc, wht, wht_bw, ih, ih_bw);
    for (int i = 1; i < seq_length; ++i) {
      int j = last - i;
      BiLstmCellLauncher(d.stream(), num_units, batch_size, tmpi + i * m * n1,
                         tmpi_bw + j * m * n1, b, oy + i * k1 * n1,
                         oy_bw + j * k1 * n1, oc, oc, wht, wht_bw,
                         oy + (i - 1) * k1 * n1, oy_bw + (j + 1) * k1 * n1);
    }

    cudaMemcpyAsync(oh, oy + (seq_length - 1) * k1 * n1,
                    k1 * n1 * sizeof(float), cudaMemcpyDeviceToDevice,
                    d.stream());

    cudaMemcpyAsync(oh_bw, oy_bw, k1 * n1 * sizeof(float),
                    cudaMemcpyDeviceToDevice, d.stream());

    YConcatLauncher(d.stream(), num_units, batch_size, seq_length, oy, oy_bw,
                    oy_concat);

    wx += dir_count * (input_size + num_units) * num_units * 4;
    wh = wx + 2 * num_units * num_units * 4;
    wx_bw = wx + (2 * num_units + num_units) * num_units * 4;
    wh_bw = wh + (2 * num_units + num_units) * num_units * 4;
    b += dir_count * num_units * 8;
    for (int layer = 1; layer < num_layers; ++layer) {
      ic += dir_count * batch_size * num_units;
      ic_bw += dir_count * batch_size * num_units;
      oc += dir_count * batch_size * num_units;
      oc_bw += dir_count * batch_size * num_units;
      ih += dir_count * batch_size * num_units;
      ih_bw += dir_count * batch_size * num_units;
      oh += dir_count * batch_size * num_units;
      oh_bw += dir_count * batch_size * num_units;
      const int k2 = 2 * num_units;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k2, &fone, wx, k2,
                  oy_concat, k2, &fzero, tmpi, m);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k2, &fone, wx_bw, k2,
                  oy_concat, k2, &fzero, tmpi_bw, m);

      cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k1, &fone, wh, k1,
                  &fzero, wh, m, wht, m);
      cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k1, &fone, wh_bw, k1,
                  &fzero, wh_bw, m, wht_bw, m);

      int last = seq_length - 1;
      BiLstmCellLauncher(d.stream(), num_units, batch_size, tmpi,
                         tmpi_bw + last * m * n1, b, oy, oy_bw + last * k1 * n1,
                         ic, oc, wht, wht_bw, ih, ih_bw);
      for (int i = 1; i < seq_length; ++i) {
        int j = last - i;
        BiLstmCellLauncher(d.stream(), num_units, batch_size, tmpi + i * m * n1,
                           tmpi_bw + j * m * n1, b, oy + i * k1 * n1,
                           oy_bw + j * k1 * n1, oc, oc, wht, wht_bw,
                           oy + (i - 1) * k1 * n1, oy_bw + (j + 1) * k1 * n1);
      }

      cudaMemcpyAsync(oh, oy + (seq_length - 1) * k1 * n1,
                      k1 * n1 * sizeof(float), cudaMemcpyDeviceToDevice,
                      d.stream());

      cudaMemcpyAsync(oh_bw, oy_bw, k1 * n1 * sizeof(float),
                      cudaMemcpyDeviceToDevice, d.stream());
      wx += dir_count * (2 * num_units + num_units) * num_units * 4;
      wx_bw += dir_count * (2 * num_units + num_units) * num_units * 4;
      wh += dir_count * (2 * num_units + num_units) * num_units * 4;
      wh_bw += dir_count * (2 * num_units + num_units) * num_units * 4;
      b += dir_count * num_units * 8;
      YConcatLauncher(d.stream(), num_units, batch_size, seq_length, oy, oy_bw,
                      oy_concat);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BladeBilstm").Device(DEVICE_GPU), BilstmOp);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
