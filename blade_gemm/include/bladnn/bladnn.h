#include <cstdint>

#pragma once
namespace bladnn {
struct Context {
  void* stream;
};

enum Dtype {
  kF32,
  kS32,
  kF16,
  kF64,
  kBF16,
  kTF32,
  kU16,
  kF8,
  kS8,
  kU8,
  kU4,
  kS4,
  kB1,
  kUnknown
};

enum Layout { kNHWC, kNCHW };
enum ConvKind { kFprop, kDgrad, kWgrad };
enum Activation { kNone, kRelu };

bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, void* c_ptr,
          int c_dim0, int c_dim1, int batch_count = 1, bool a_is_const = false,
          bool b_is_const = false, const void* alpha = nullptr,
          const void* beta = nullptr);

// a must be sparse tensor
bool spgemm(Context* ctx, Dtype a_dtype, bool a_transpose, const void* a_ptr,
            int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
            const void* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype,
            void* c_ptr, int c_dim0, int c_dim1, Dtype e_dtype,
            const void* e_ptr, int e_dim0, int e_dim1, int batch_count = 1,
            bool a_is_const = false, bool b_is_const = false,
            const void* alpha = nullptr, const void* beta = nullptr);

bool conv2d(void* s, Dtype in_dtype, Dtype out_dtype, ConvKind conv_kind,
            Layout data_layout, Layout kernel_layout, int N, int H, int W,
            int C, int K, int R, int S, int P, int Q, int pad_h, int pad_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int groups,

            void const* alpha,  /// Pointer to alpha scalar

            void const* ptr_A,  /// Pointer to A matrix in Global Memory

            void const* ptr_B,  /// Pointer to B matrix in Global Memory

            void const* beta,  /// Pointer to beta scalar

            void const* ptr_C,  /// Pointer to C matrix

            void* ptr_D,  /// Pointer to D matrix

            bool bias_c = false,

            Activation activation = Activation::kNone);

// definition for per-channel quantization gemm
// D = scale(A * B + alpha * C)
bool gemm(Context* ctx, Dtype a_dtype, bool a_transpose, int8_t* a_ptr,
          int a_dim0, int a_dim1, Dtype b_dtype, bool b_transpose,
          int8_t* b_ptr, int b_dim0, int b_dim1, Dtype c_dtype, int8_t* d_ptr,
          int c_dim0, int c_dim1, int batch_count, bool a_is_const,
          bool b_is_const, float* alpha, float* beta, float* scale, float* bias,
          int8_t* c_ptr = nullptr);

// definition for per-channel quantization conv2d
bool conv2d(Context* ctx, Dtype in_dtype, Dtype out_dtype, ConvKind conv_kind,
            Layout data_layout, Layout kernel_layout, int N, int H, int W,
            int C, int K, int R, int S, int P, int Q, int pad_h, int pad_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int groups, int8_t* ptr_A, int8_t* ptr_B, int8_t* ptr_D,
            float* alpha = nullptr, float* beta = nullptr,
            float* scale = nullptr, float* bias = nullptr,
            int8_t* ptr_C = nullptr);
}  // namespace bladnn
