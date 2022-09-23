// RUN: disc-opt -disc-dense-to-sparse=enable-sparse-convert=false -split-input-file %s -o - | FileCheck %s --check-prefix=DenseToSparse
// RUN: disc-opt -disc-sparse-gemm-transpose-simplifier -split-input-file %s -o - | FileCheck %s --check-prefix=SparseGemmTransSim

// DenseToSparse-LABEL: func.func @dense_to_sparse_gemm
func.func @dense_to_sparse_gemm(%arg0: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %0 = mhlo.constant dense<[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]> : tensor<4x4xf16>
  %1 = "mhlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  // DenseToSparse: mhlo.transpose
  // DenseToSparse-NOT: mhlo.dot_general
  // DenseToSparse: mhlo_disc.custom_call
  return %1: tensor<4x4xf16>
}

// SparseGemmTransSim-LABEL: func.func @sparse_gemm_transpose_simplifier
func.func @sparse_gemm_transpose_simplifier(%arg0: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %0 = mhlo.constant dense<[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]> : tensor<4x4xf16>
  %1 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x4xf16>) -> tensor<4x4xf16>
  %2 = "mhlo_disc.custom_call"(%1, %0) {backend_config = {lhs_contracting_dimensions = 0 : i64, rhs_contracting_dimensions = 0 : i64}, call_target_name = "sparse_gemm", has_side_effect = false} : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  // SparseGemmTransSim-NOT: mhlo.transpose
  return %2: tensor<4x4xf16>
}