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

#include <stdio.h>

#include <iostream>
#include <sstream>
#include <thread>

#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/context/stream_executor_based_impl.h"
#include "mlir/ral/device/gpu/gpu_driver.h"

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
#include "bladnn/bladnn.h"
#endif

namespace tao {
namespace ral {

namespace gpu {

namespace se = ::stream_executor;

namespace se_impl {

template <typename WeightDtype, typename MetaDtype>
bool generate_sparse_balance_weight(bool w_transpose, int w_dim0, int w_dim1,
                                    WeightDtype* ptr_W,
                                    WeightDtype* ptr_compress_W,
                                    MetaDtype* ptr_M, MetaDtype* ptr_M_buf) {
  // Process 4bit meta data a time
  int step;

  // 1:2 or 2:4 or 4:8
  int m, n;

  if (sizeof(WeightDtype) == 0.5) {
    // int4
    step = 8;
    m = 4;
    n = 8;
  } else if (sizeof(WeightDtype) == 1) {
    // int8
    step = 4;
    m = 2;
    n = 4;
  } else if (sizeof(WeightDtype) == 2) {
    // float16
    step = 4;
    m = 2;
    n = 4;
  } else if (sizeof(WeightDtype) == 4) {
    // float32
    step = 2;
    m = 1;
    n = 2;
  }

  int sparse_element = 2;
  int element_per_m = 16 / std::log2(n);
  int decom_element_per_m = 32 / sizeof(WeightDtype);
  int row = w_dim0;
  int col = w_dim1;

  // int ElementsPerM = (sizeof(WeightDtype) == 0.5) ? 2 : 1;
  for (int r = 0; r < row; ++r) {
    int w_count = 0;
    for (int c = 0; c < (col / decom_element_per_m); ++c) {
      std::vector<int> unremove_indices;
      for (int i = 0; i < decom_element_per_m; i++) {
        long long int a_index = 0;
        if (w_transpose) {
          a_index = (c * decom_element_per_m + i) * row + r;
        } else {
          a_index = r * col + c * decom_element_per_m + i;
        }
        if (static_cast<float>(ptr_W[a_index]) != 0) {
          if (w_transpose) {
            ptr_compress_W[w_count * row + r] = ptr_W[a_index];
          } else {
            ptr_compress_W[r * col / sparse_element + w_count] = ptr_W[a_index];
          }
          unremove_indices.push_back(i % n);
          w_count++;
        }
      }
      int e_indices = r * col / decom_element_per_m + c;
      ptr_M_buf[e_indices] = 0;
      for (int i = 0; i < unremove_indices.size(); ++i) {
        ptr_M_buf[e_indices] |= (unremove_indices[i] << (2 * i));
      }
    }
  }

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col / sparse_element / element_per_m; j++) {
      int group = (sizeof(MetaDtype) == 2) ? 32 : 16;
      int interweave = (sizeof(MetaDtype) == 2) ? 4 : 2;

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

      ptr_M[dest_col_major * row * 2 + dest_row * 2 + dest_col_minor] =
          ptr_M_buf[i * col / sparse_element / element_per_m + j];
    }
  }

  return true;
}

struct GpuSparseGemmKey {
  int m = -1;
  int n = -1;
  int k = -1;
  int batch = 1;
  bool transpose_a = false;
  bool transpose_b = false;
  opaque_t const_weight_ptr = nullptr;

  bool operator==(const GpuSparseGemmKey& rhs) const {
    return m == rhs.m && n == rhs.n && k == rhs.k && batch == rhs.batch &&
           transpose_a == rhs.transpose_a && transpose_b == rhs.transpose_b &&
           const_weight_ptr == rhs.const_weight_ptr;
  }
};

template <typename WeightDtype, typename MetaDtype>
class SparseCompressedWeight {
 public:
  SparseCompressedWeight(ExecutionContext* ctx, void* stream_handle,
                         const GpuSparseGemmKey& key);
  ~SparseCompressedWeight();

  opaque_t compressed_weight() const { return compressed_weight_; }
  opaque_t compressed_meta() const { return compressed_meta_; }

 private:
  ExecutionContext* ctx_;
  gpu::GPUDriver* driver_;
  void* stream_;
  opaque_t compressed_weight_ = nullptr;
  opaque_t compressed_meta_ = nullptr;
};

template <typename WeightDtype, typename MetaDtype>
SparseCompressedWeight<WeightDtype, MetaDtype>::SparseCompressedWeight(
    ExecutionContext* ctx, void* stream_handle, const GpuSparseGemmKey& key)
    : ctx_(ctx),
      driver_(ctx->getDriver<gpu::GPUDriver>(gpu::GPUDriver::name())),
      stream_(stream_handle) {
  std::vector<WeightDtype> host_B(key.m * key.k);
  std::vector<WeightDtype> host_compressed_B(key.m * key.k / 2);
  std::vector<MetaDtype> host_compressed_E(key.m * key.k / 16);
  std::vector<MetaDtype> host_compressed_reorder_E(key.m * key.k / 16);
  driver_->d2h(ctx_, stream_, (void*)key.const_weight_ptr, (void*)host_B.data(),
               key.m * key.k * sizeof(WeightDtype));

  generate_sparse_balance_weight<WeightDtype, MetaDtype>(
      !key.transpose_b, key.m, key.k, host_B.data(), host_compressed_B.data(),
      host_compressed_reorder_E.data(), host_compressed_E.data());

  compressed_weight_ =
      driver_->alloc_persistent(ctx_, key.m * key.k / 2 * sizeof(WeightDtype));
  compressed_meta_ =
      driver_->alloc_persistent(ctx_, key.m * key.k / 16 * sizeof(MetaDtype));

  driver_->h2d(ctx_, stream_, (void*)host_compressed_B.data(),
               static_cast<void*>(compressed_weight_),
               key.m * key.k / 2 * sizeof(WeightDtype));
  driver_->h2d(ctx_, stream_, (void*)host_compressed_reorder_E.data(),
               static_cast<void*>(compressed_meta_),
               key.m * key.k / 16 * sizeof(MetaDtype));

  driver_->syncOnStream(ctx_, stream_);
}

template <typename WeightDtype, typename MetaDtype>
SparseCompressedWeight<WeightDtype, MetaDtype>::~SparseCompressedWeight() {}

template <typename WeightDtype, typename MetaDtype>
struct SparseGemmState : public Context::Resource {
  std::mutex mu;
  std::map<intptr_t,
           std::shared_ptr<SparseCompressedWeight<WeightDtype, MetaDtype>>>
      cache;
};

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
template <typename T>
bladnn::Dtype toBlaDNNDtype() {
  if (std::is_same<T, Eigen::half>::value) {
    return bladnn::Dtype::kF16;
  }
  if (std::is_same<T, float>::value) {
    return bladnn::Dtype::kF32;
  }
  if (std::is_same<T, double>::value) {
    return bladnn::Dtype::kF64;
  }
  if (std::is_same<T, uint16_t>::value) {
    return bladnn::Dtype::kU16;
  }
  return bladnn::Dtype::kUnknown;
}
#endif

template <typename InT, typename OutT, typename MetaT>
void ral_spgemm(ExecutionContext* ctx, void* stream_handle,
                MemRefType<InT, 2> A, MemRefType<InT, 2> B,
                MemRefType<OutT, 2> C, int cd_a, int cd_b) {
  InT* dA = A.data;
  InT* dB = B.data;
  OutT* dC = C.data;

  int64_t num_A_rows = A.sizes[0];
  int64_t num_A_cols = A.sizes[1];
  int64_t num_B_rows = B.sizes[0];
  int64_t num_B_cols = B.sizes[1];

  bool tp_a = cd_a == 1 ? false : true;
  bool tp_b = cd_b == 0 ? false : true;

  int64_t m = tp_b ? num_B_rows : num_B_cols;
  int64_t n = tp_a ? num_A_cols : num_A_rows;
  int64_t k = tp_b ? num_B_cols : num_B_rows;

  float alpha = 1.0f;
  float beta = 0.0f;

  std::string unique_name =
      "tao_ral.gpu.sparse_gemm_" + tao::ral::TaoTypeNameHelper<InT>::Invoke();
  auto state = ctx->getOrCreateResource<SparseGemmState<InT, MetaT>>(
      unique_name, []() { return new SparseGemmState<InT, MetaT>; });
  opaque_t compressed_weight, compressed_meta;
  {
    GpuSparseGemmKey key{m, n, k, 1, tp_a, tp_b, dB};
    std::lock_guard<std::mutex> l(state->mu);

    auto& cache = state->cache;
    auto it = cache.find((intptr_t)dB);
    if (it == cache.end()) {
      std::shared_ptr<SparseCompressedWeight<InT, MetaT>> compressed_weight_ptr(
          new SparseCompressedWeight<InT, MetaT>(ctx, stream_handle, key));
      it = cache.insert(std::make_pair((intptr_t)dB, compressed_weight_ptr))
               .first;
    }
    compressed_weight = it->second->compressed_weight();
    compressed_meta = it->second->compressed_meta();
  }

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  {
    auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
    void* s = gpu_driver->asCUStream(ctx, stream_handle);
    bladnn::Context bladnn_ctx{s};
    bladnn::Dtype in_dtype = toBlaDNNDtype<InT>();
    bladnn::Dtype out_dtype = toBlaDNNDtype<OutT>();
    bladnn::Dtype meta_dtype = toBlaDNNDtype<MetaT>();
    bool ret = bladnn::spgemm(
        &bladnn_ctx, in_dtype, !tp_b, compressed_weight, tp_b ? m : k / 2,
        tp_b ? k / 2 : m, in_dtype, !tp_a, dA, tp_a ? k : n, tp_a ? n : k,
        out_dtype, dC, m, n, meta_dtype, compressed_meta, k / 32, m * 2);

    if (ret) {
      return;
    } else {
      ctx->signalError(Context::FAILURE, "run sparse gemm failed.");
    }
  }
#else
  ctx->signalError(Context::FAILURE, "unsupport sparse gemm.");
#endif
}

template <typename InT, typename OutT, typename MetaT>
void ral_spgemm_balance(ExecutionContext* ctx, void* stream_handle,
                        MemRefType<InT, 2> A, MemRefType<InT, 2> B_data,
                        MemRefType<MetaT, 1> B_indices, MemRefType<OutT, 2> C,
                        int cd_a, int cd_b) {
  InT* dA = A.data;
  InT* dB_data = B_data.data;
  MetaT* dB_indices = B_indices.data;
  OutT* dC = C.data;

  int64_t num_A_rows = A.sizes[0];
  int64_t num_A_cols = A.sizes[1];
  int64_t num_B_rows = B_data.sizes[0];
  int64_t num_B_cols = B_data.sizes[1];

  bool tp_a = cd_a == 1 ? false : true;
  bool tp_b = cd_b == 0 ? false : true;

  int64_t m = tp_b ? num_B_rows : num_B_cols;
  int64_t n = tp_a ? num_A_cols : num_A_rows;
  int64_t k = tp_a ? num_A_rows : num_A_cols;

  float alpha = 1.0f;
  float beta = 0.0f;

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  {
    auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
    void* s = gpu_driver->asCUStream(ctx, stream_handle);
    bladnn::Context bladnn_ctx{s};
    bladnn::Dtype in_dtype = toBlaDNNDtype<InT>();
    bladnn::Dtype out_dtype = toBlaDNNDtype<OutT>();
    bladnn::Dtype meta_dtype = toBlaDNNDtype<MetaT>();
    bool ret = bladnn::spgemm(&bladnn_ctx, in_dtype, !tp_b, dB_data,
                              tp_b ? m : k / 2, tp_b ? k / 2 : m, in_dtype,
                              !tp_a, dA, tp_a ? k : n, tp_a ? n : k, out_dtype,
                              dC, m, n, meta_dtype, dB_indices, k / 32, m * 2);

    if (ret) {
      return;
    } else {
      ctx->signalError(Context::FAILURE, "run sparse gemm failed.");
    }
  }
#else
  ctx->signalError(Context::FAILURE, "unsupport sparse gemm.");
#endif
}

}  // namespace se_impl
}  // namespace gpu

TAO_RAL_API("sparse_gemm", "gpu",
            gpu::se_impl::ral_spgemm<Eigen::half, Eigen::half, uint16_t>);

TAO_RAL_API(
    "sparse_gemm", "gpu",
    gpu::se_impl::ral_spgemm_balance<Eigen::half, Eigen::half, uint16_t>);

}  // namespace ral
}  // namespace tao