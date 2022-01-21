// RUN: disc-opt -disc-dot-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @dot_transpose_2d
func @dot_transpose_2d(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [0]
  return %1: tensor<?x?xf32>
}

// CHECK-LABEL: func @dot_transpose_3d
func @dot_transpose_3d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_batching_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  return %1: tensor<?x?x?xf32>
}

// CHECK-LABEL: func @dot_transpose_batching_3d
func @dot_transpose_batching_3d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_batching_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [2]
  return %1: tensor<?x?x?xf32>
}
