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

#include "bilstm_op_kernels.cu.h"

namespace tensorflow {

#define BI_LSTM_CeLL(B)                                                       \
  BiLstmCell<float, B, BX, BY><<<dim3(GX, GY), dim3(BX, BY), 0, stream>>>(    \
      hidden, batch, tmp_i, tmp_i_bw, bias, h_out, h_out_bw, c_in, c_out, wh, \
      h_in, wh_bw, h_in_bw);

void BiLstmCellLauncher(cudaStream_t stream, const int hidden, const int batch,
                        const float* tmp_i, const float* tmp_i_bw,
                        const float* bias, float* h_out, float* h_out_bw,
                        const float* c_in, float* c_out, const float* wh,
                        const float* wh_bw, const float* h_in,
                        const float* h_in_bw) {
  const int BX = 32;
  const int BY = 16;  // BY >= batch
  const int GY = 2;
  const int GX = (hidden + BX - 1) / BX;
  switch (batch) {
    case 1:
      BI_LSTM_CeLL(1);
      break;
    case 2:
      BI_LSTM_CeLL(2);
      break;
    case 3:
      BI_LSTM_CeLL(3);
      break;
    case 4:
      BI_LSTM_CeLL(4);
      break;
    case 5:
      BI_LSTM_CeLL(5);
      break;
    case 6:
      BI_LSTM_CeLL(6);
      break;
    case 7:
      BI_LSTM_CeLL(7);
      break;
    case 8:
      BI_LSTM_CeLL(8);
      break;
    case 9:
      BI_LSTM_CeLL(9);
      break;
    case 10:
      BI_LSTM_CeLL(10);
      break;
    case 11:
      BI_LSTM_CeLL(11);
      break;
    case 12:
      BI_LSTM_CeLL(12);
      break;
    case 13:
      BI_LSTM_CeLL(13);
      break;
    case 14:
      BI_LSTM_CeLL(14);
      break;
    case 15:
      BI_LSTM_CeLL(15);
      break;
    case 16:
      BI_LSTM_CeLL(16);
      break;
    case 17:
      BI_LSTM_CeLL(17);
      break;
    case 18:
      BI_LSTM_CeLL(18);
      break;
    case 19:
      BI_LSTM_CeLL(19);
      break;
    case 20:
      BI_LSTM_CeLL(20);
      break;
    case 21:
      BI_LSTM_CeLL(21);
      break;
    case 22:
      BI_LSTM_CeLL(22);
      break;
    case 23:
      BI_LSTM_CeLL(23);
      break;
    case 24:
      BI_LSTM_CeLL(24);
      break;
    case 25:
      BI_LSTM_CeLL(25);
      break;
    case 26:
      BI_LSTM_CeLL(26);
      break;
    case 27:
      BI_LSTM_CeLL(27);
      break;
    case 28:
      BI_LSTM_CeLL(28);
      break;
    case 29:
      BI_LSTM_CeLL(29);
      break;
    case 30:
      BI_LSTM_CeLL(30);
      break;
    case 31:
      BI_LSTM_CeLL(31);
      break;
    case 32:
      BI_LSTM_CeLL(32);
      break;
  }
  // TODO(minmin): add more batch sizes
}

// void LstmCellLauncher(cudaStream_t stream, const int hidden, const int batch,
//                       const float* tmp_i, const float* bias, float* h_out,
//                       const float* c_in, float* c_out, const float* wh,
//                       const float* h_in) {
//   const int BX = 32;
//   const int BY = 32;
//   const int GY = 1;
//   const int GX = (hidden + BX - 1) / BX;
//   if (batch == 1)
//     LstmCell<float, 1, BX, BY><<<dim3(GX, GY), dim3(BX, BY), 0, stream>>>(
//         hidden, batch, tmp_i, bias, h_out, c_in, c_out, wh, h_in);
//   // TODO(minmin): add more batch sizes
// }

void YConcatLauncher(cudaStream_t stream, const int hidden, const int batch,
                     const int length, const float* oy, const float* oy_bw,
                     float* oy_concat) {
  YConcat<float><<<dim3(batch * length, 2), 32, 0, stream>>>(hidden, oy, oy_bw,
                                                             oy_concat);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
