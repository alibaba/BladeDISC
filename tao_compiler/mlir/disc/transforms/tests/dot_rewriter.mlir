// RUN: disc-opt -disc-dot-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @dot_transpose_2d
func.func @dot_transpose_2d(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [0]
  return %1: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dot_transpose_3d
func.func @dot_transpose_3d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_batching_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [1]
  return %1: tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_transpose_batching_3d
func.func @dot_transpose_batching_3d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_batching_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: mhlo.transpose
  // CHECK: mhlo.dot_general
  // CHECK: lhs_contracting_dimensions = [2]
  return %1: tensor<?x?x?xf32>
}

// CHECK-LABEL:  func.func @dot_2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[256, 1]> : tensor<2xindex>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG1]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<256x1xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x1xf32>
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]] : tensor<1xindex>
// CHECK:         %[[T4:.*]] = mhlo.dynamic_reshape %[[T1]], %[[T3]] : (tensor<?x1xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:         return %[[T4]] : tensor<?xf32>
func.func @dot_2dx1d(%arg0: tensor<?x256xf32>, %arg1: tensor<256xf32>) -> tensor<?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x256xf32>, tensor<256xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL:  func.func @dot_1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 256]> : tensor<2xindex>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[T0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x?xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<1x?xf32>
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]] : tensor<1xindex>
// CHECK:         %[[T4:.*]] = mhlo.dynamic_reshape %[[T1]], %[[T3]] : (tensor<1x?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:         return %[[T4]] : tensor<?xf32>
func.func @dot_1dx2d(%arg0: tensor<256xf32>, %arg1: tensor<256x?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<256xf32>, tensor<256x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL:  func.func @dot_1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<f32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 256]> : tensor<2xindex>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<[256, 1]> : tensor<2xindex>
// CHECK:         %[[CST_1:.*]] = arith.constant dense<1> : tensor<1xindex>
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.dynamic_reshape %[[ARG1]], %[[CST_0]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<256x1xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:         %[[T3:.*]] = mhlo.dynamic_reshape %[[T2]], %[[CST_1]] : (tensor<1x1xf32>, tensor<1xindex>) -> tensor<1xf32>
// CHECK:         %[[T4:.*]] = mhlo.reshape %[[T3]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         return %[[T4]] : tensor<f32>
func.func @dot_1dx1d(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<f32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL:  func.func @dot_general_1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 256]> : tensor<2xindex>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[T0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x?xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<1x?xf32>
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]] : tensor<1xindex>
// CHECK:         %[[T4:.*]] = mhlo.dynamic_reshape %[[T1]], %[[T3]] : (tensor<1x?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:         return %[[T4]] : tensor<?xf32>
func.func @dot_general_1dx2d(%arg0: tensor<256xf32>, %arg1: tensor<256x?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<256xf32>, tensor<256x?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL:  func.func @dot_general_2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[256, 1]> : tensor<2xindex>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG1]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<256x1xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x1xf32>
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]] : tensor<1xindex>
// CHECK:         %[[T4:.*]] = mhlo.dynamic_reshape %[[T1]], %[[T3]] : (tensor<?x1xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:         return %[[T4]] : tensor<?xf32>
func.func @dot_general_2dx1d(%arg0: tensor<?x256xf32>, %arg1: tensor<256xf32>) -> tensor<?xf32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL:  func.func @dot_general_1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<f32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 256]> : tensor<2xindex>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<[256, 1]> : tensor<2xindex>
// CHECK:         %[[CST_1:.*]] = arith.constant dense<1> : tensor<1xindex>
// CHECK:         %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[CST]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.dynamic_reshape %[[ARG1]], %[[CST_0]] : (tensor<256xf32>, tensor<2xindex>) -> tensor<256x1xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:         %[[T3:.*]] = mhlo.dynamic_reshape %[[T2]], %[[CST_1]] : (tensor<1x1xf32>, tensor<1xindex>) -> tensor<1xf32>
// CHECK:         %[[T4:.*]] = mhlo.reshape %[[T3]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         return %[[T4]] : tensor<f32>
func.func @dot_general_1dx1d(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<f32> {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----
// CHECK-LABEL: func.func @dot_general_transpose(
// CHECK-SAME:        %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:         return %[[T0]]
// CHECK:       }
func.func @dot_general_transpose(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%1, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}
