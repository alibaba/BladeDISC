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

#ifndef DYN_SORT_GATHER_TOP_K_H_
#define DYN_SORT_GATHER_TOP_K_H_

template <typename Dtype, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void gatherTopK2DKernel(const Dtype* input, Dtype* output,
                            const unsigned in_k, const unsigned out_k,
                            const unsigned batch) {
  for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < (out_k * batch); idx += blockDim.x * gridDim.x) {
    unsigned rd_idx = (idx / out_k) * in_k + (idx % out_k);
    output[idx] = input[rd_idx];
  }
}

template <typename Dlhs, typename Drhs, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void gatherTopK2DPairKernel(const Dlhs* in_lhs, const Drhs* in_rhs,
                                Dlhs* out_lhs, Drhs* out_rhs,
                                const unsigned in_k, const unsigned out_k,
                                const unsigned batch) {
  for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < (out_k * batch); idx += blockDim.x * gridDim.x) {
    unsigned rd_idx = (idx / out_k) * in_k + (idx % out_k);
    out_lhs[idx] = in_lhs[rd_idx];
    out_rhs[idx] = in_rhs[rd_idx];
  }
}

template <typename Dtype>
void gatherTopK2D(const Dtype* input, Dtype* output, const unsigned in_k,
                  const unsigned out_k, const unsigned batch,
                  cudaStream_t stream) {
  unsigned n = out_k * batch;
  if (n < 32) {
    gatherTopK2DKernel<Dtype, 32>
        <<<1, 32, 0, stream>>>(input, output, in_k, out_k, batch);
  } else if (n < 128) {
    gatherTopK2DKernel<Dtype, 128>
        <<<1, 128, 0, stream>>>(input, output, in_k, out_k, batch);
  } else if (n < 256) {
    gatherTopK2DKernel<Dtype, 256>
        <<<1, 256, 0, stream>>>(input, output, in_k, out_k, batch);
  } else {
    unsigned num_blocks = (n + 256 - 1) / 256;
    gatherTopK2DKernel<Dtype, 256>
        <<<num_blocks, 256, 0, stream>>>(input, output, in_k, out_k, batch);
  }
}

template <typename Dlhs, typename Drhs>
void gatherTopK2DPair(const Dlhs* in_lhs, const Drhs* in_rhs, Dlhs* out_lhs,
                      Drhs* out_rhs, const unsigned in_k, const unsigned out_k,
                      const unsigned batch, cudaStream_t stream) {
  unsigned n = out_k * batch;
  if (n < 32) {
    gatherTopK2DPairKernel<Dlhs, Drhs, 32><<<1, 32, 0, stream>>>(
        in_lhs, in_rhs, out_lhs, out_rhs, in_k, out_k, batch);
  } else if (n < 128) {
    gatherTopK2DPairKernel<Dlhs, Drhs, 128><<<1, 128, 0, stream>>>(
        in_lhs, in_rhs, out_lhs, out_rhs, in_k, out_k, batch);
  } else if (n < 256) {
    gatherTopK2DPairKernel<Dlhs, Drhs, 256><<<1, 256, 0, stream>>>(
        in_lhs, in_rhs, out_lhs, out_rhs, in_k, out_k, batch);
  } else {
    unsigned num_blocks = (n + 256 - 1) / 256;
    gatherTopK2DPairKernel<Dlhs, Drhs, 256><<<num_blocks, 256, 0, stream>>>(
        in_lhs, in_rhs, out_lhs, out_rhs, in_k, out_k, batch);
  }
}

#endif  // DYN_SORT_GATHER_TOP_K_H_
