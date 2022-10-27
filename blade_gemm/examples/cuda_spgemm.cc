#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "bladnn/bladnn.h"
#include "cuda_fp16.h"

template <typename ElementA, typename ElementE>
bool generate_sparse_weight(bool a_transpose, int a_dim0, int a_dim1,
                            ElementA* ptr_A, ElementA* ptr_compress_A,
                            ElementE* ptr_E, ElementE* ptr_E_buf) {
  // Process 4bit meta data a time
  int step;

  // 1:2 or 2:4 or 4:8
  int m, n;

  if (sizeof(ElementA) == 0.5) {
    // int4
    step = 8;
    m = 4;
    n = 8;
  } else if (sizeof(ElementA) == 1) {
    // int8
    step = 4;
    m = 2;
    n = 4;
  } else if (sizeof(ElementA) == 2) {
    // float16
    step = 4;
    m = 2;
    n = 4;
  } else if (sizeof(ElementA) == 4) {
    // float32
    step = 2;
    m = 1;
    n = 2;
  }

  int sparse_element = 2;
  int element_per_e = 16 / std::log2(n);
  int decom_element_per_e = 32 / sizeof(ElementA);
  int row = a_dim0;
  int col = a_dim1;

  int ElementsPerE = (sizeof(ElementA) == 0.5) ? 2 : 1;
  for (int r = 0; r < row; ++r) {
    int a_count = 0;
    for (int c = 0; c < (col / decom_element_per_e); ++c) {
      std::vector<int> unremove_indices;
      for (int i = 0; i < decom_element_per_e; i++) {
        int a_index = 0;
        if (a_transpose) {
          a_index = (c * decom_element_per_e + i) * row + r;
        } else {
          a_index = r * col + c * decom_element_per_e + i;
        }
        if (ptr_A[a_index] != 0) {
          if (a_transpose) {
            ptr_compress_A[a_count * row + r] = ptr_A[a_index];
          } else {
            ptr_compress_A[r * col / sparse_element + a_count] = ptr_A[a_index];
          }
          unremove_indices.push_back(i % n);
          a_count++;
        }
      }
      int e_indices = r * col / decom_element_per_e + c;
      ptr_E_buf[e_indices] = 0;
      for (int i = 0; i < unremove_indices.size(); ++i) {
        ptr_E_buf[e_indices] |= (unremove_indices[i] << (2 * i));
      }
    }
  }

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col / sparse_element / element_per_e; j++) {
      int group = (sizeof(ElementE) == 2) ? 32 : 16;
      int interweave = (sizeof(ElementE) == 2) ? 4 : 2;

      int dest_row = i / group * group + (i % 8) * interweave + (i % group) / 8;
      int dest_col = j;

      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      int dest_col_major = dest_col / 2;
      int dest_col_minor = dest_col % 2;

      ptr_E[dest_col_major * row * 2 + dest_row * 2 + dest_col_minor] =
          ptr_E_buf[i * col / sparse_element / element_per_e + j];
    }
  }

  return true;
}

bool spgemm(bladnn::Context& bladnn_ctx, bladnn::Dtype a_dtype,
            bladnn::Dtype e_dtype, void* a, void* b, void* c, void* e, int M,
            int N, int K, bool tp_a, bool tp_b) {
  // when using bladnn::spgemm, a must be sparse tensor
  bool ret =
      bladnn::spgemm(&bladnn_ctx, a_dtype, tp_a, a, tp_a ? K / 2 : M,
                     tp_a ? M : K / 2, a_dtype, tp_b, b, tp_b ? N : K,
                     tp_b ? K : N, a_dtype, c, M, N, e_dtype, e, K / 32, M * 2);
  return ret;
}

bool gemm(bladnn::Context& bladnn_ctx, bladnn::Dtype dtype, void* a, void* b,
          void* c, int M, int N, int K, bool tp_a, bool tp_b) {
  bool ret =
      bladnn::gemm(&bladnn_ctx, dtype, tp_a, a, tp_a ? K : M, tp_a ? M : K,
                   dtype, tp_b, b, tp_b ? N : K, tp_b ? K : N, dtype, c, M, N);
  return ret;
}

bool hspgemm() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  bladnn::Context bladnn_ctx{s};
  bladnn::Dtype a_dtype = bladnn::Dtype::kF16;
  bladnn::Dtype e_dtype = bladnn::Dtype::kU16;
  bool a_trans = false;
  bool b_trans = false;

  void *a, *a_com, *b, *c_sparse, *c_dense, *e;
  int M = 768;
  int N = 800;
  int K = 3072;

  // generate sparse weight
  std::vector<__half> ha(M * K);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K / 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        int a_index = 0;
        if (a_trans)
          a_index = (j * 4 + k) * M + i;
        else
          a_index = i * K + j * 4 + k;

        if (k % 2 == 0) {
          ha[a_index] = rand() * 1.0 / RAND_MAX * 2.0f - 1.0f;
        } else {
          ha[a_index] = 0.0f;
        }
      }
    }
  }

  std::vector<__half> hb(K * N);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      hb[i * N + j] = rand() * 1.0 / RAND_MAX * 2.0f - 1.0f;
    }
  }

  std::vector<__half> ha_compressed(M * K / 2);
  std::vector<uint16_t> he(M * K / 2 / 8);
  std::vector<uint16_t> he_buf(M * K / 2 / 8);
  auto result = generate_sparse_weight<__half, uint16_t>(
      a_trans, M, K, ha.data(), ha_compressed.data(), he.data(), he_buf.data());

  cudaMalloc(&c_sparse, M * N * 2);
  cudaMalloc(&c_dense, M * N * 2);
  cudaMalloc(&a, M * K * 2);
  cudaMalloc(&a_com, M * K * 2 / 2);
  cudaMalloc(&b, K * N * 2);
  cudaMalloc(&e, K * N * 2 / 2 / 8);

  cudaMemcpy(static_cast<__half*>(a), ha.data(), M * K * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<__half*>(a_com), ha_compressed.data(),
             M * K / 2 * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<__half*>(b), hb.data(), K * N * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<uint16_t*>(e), he.data(),
             M * K / 2 / 8 * sizeof(uint16_t), cudaMemcpyHostToDevice);

  bool ret = true;
  ret &= spgemm(bladnn_ctx, a_dtype, e_dtype, a_com, b, c_sparse, e, M, N, K,
                a_trans, b_trans);
  if (!ret) {
    std::cout << "run sparse gemm failed" << std::endl;
    return ret;
  }

  ret &= gemm(bladnn_ctx, a_dtype, a, b, c_dense, M, N, K, a_trans, b_trans);
  if (!ret) {
    std::cout << "run dense gemm failed" << std::endl;
    return ret;
  }

  std::vector<__half> hc_dense(M * N * 2);
  std::vector<__half> hc_sparse(M * N * 2);
  cudaMemcpy(hc_dense.data(), static_cast<__half*>(c_dense),
             M * N * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hc_sparse.data(), static_cast<__half*>(c_sparse),
             M * N * sizeof(__half), cudaMemcpyDeviceToHost);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < 10; ++j) {
      if (static_cast<float>(hc_dense[i * N + j]) -
              static_cast<float>(hc_sparse[i * N + j]) >
          1e-2) {
        std::cout << "error value :(" << i << "," << j
                  << "), dense=" << static_cast<float>(hc_dense[i * N + j])
                  << ", sparse=" << static_cast<float>(hc_sparse[i * N + j])
                  << std::endl;
      }
    }
  }

  return ret;
}

int main() {
  hspgemm();
  return 0;
}
