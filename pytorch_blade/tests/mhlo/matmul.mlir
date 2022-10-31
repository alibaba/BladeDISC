// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.mm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x3xf32>) -> tensor<2x3xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
// CHECK:         return %[[T0]] : tensor<2x3xf32>
func.func @torch.aten.mm(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.bmm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<10x3x4xf32>, %[[ARG1:.*]]: tensor<10x4x5xf32>) -> tensor<10x3x5xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<10x3x4xf32>, tensor<10x4x5xf32>) -> tensor<10x3x5xf32>
// CHECK:         return %[[T0]] : tensor<10x3x5xf32>
func.func @torch.aten.bmm(%arg0: !torch.vtensor<[10,3,4],f32>, %arg1: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,3,4],f32>, !torch.vtensor<[10,4,5],f32> -> !torch.vtensor<[10,3,5],f32>
  return %0 : !torch.vtensor<[10,3,5],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.bmm.dyn(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x?x256xf32>, %[[ARG1:.*]]: tensor<4x256x?xf32>) -> tensor<4x?x?xf32> {
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<4x256x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.from_elements %[[C4_I32]], %[[C256_I32]], %[[T1]] : tensor<3xi32>
// CHECK:         %[[T3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[T2]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x256x?xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK:         %[[T4:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T3]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x?x256xf32>, tensor<4x256x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         return %[[T4]] : tensor<4x?x?xf32>
func.func @torch.aten.bmm.dyn(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[4,256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.dyn(
// CHECK:         arith.index_cast
// CHECK:         tensor.from_elements
// CHECK:         mhlo.dynamic_reshape
// CHECK:         mhlo.dot
// CHECK:         mhlo.dynamic_reshape
// CHECK-NOT:     mhlo.dynamic_broadcast_in_dim
// CHECK-NOT:     mhlo.dot_general
func.func @torch.aten.matmul.dyn(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256x120xf32>, %[[ARG1:.*]]: tensor<4x120x256xf32>) -> tensor<4x256x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.broadcast_in_dim"(%[[ARG0]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<256x120xf32>) -> tensor<4x256x120xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[T0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x256x120xf32>, tensor<4x120x256xf32>) -> tensor<4x256x256xf32>
// CHECK:         return %[[T1]] : tensor<4x256x256xf32>
func.func @torch.aten.matmul(%arg0: !torch.vtensor<[256,120],f32>, %arg1: !torch.vtensor<[4,120,256],f32>) -> !torch.vtensor<[4,256,256],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256,120],f32>, !torch.vtensor<[4,120,256],f32> -> !torch.vtensor<[4,256,256],f32>
  return %0 : !torch.vtensor<[4,256,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.3dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1x?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<1x?xf32> {
// CHECK:         %[[T0:.*]] = mhlo.reshape %[[ARG1]] : (tensor<256xf32>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x?x256xf32>, tensor<1x256xf32>) -> tensor<1x?xf32>
// CHECK:         return %[[T1]] : tensor<1x?xf32>
func.func @torch.aten.matmul.3dx1d(%arg0: !torch.vtensor<[1,?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[1,?],f32>
  return %0 : !torch.vtensor<[1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.1dx3d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<?x256x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x256x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.from_elements %[[T1]], %[[C256_I32]] : tensor<2xi32>
// CHECK:         %[[T3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[T2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>, tensor<2xi32>) -> tensor<?x256xf32>
// CHECK:         %[[T4:.*]] = "mhlo.dot_general"(%[[T3]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x256xf32>, tensor<?x256x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T4]] : tensor<?x?xf32>
func.func @torch.aten.matmul.1dx3d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[?,256,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<?xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) : (tensor<?x256xf32>, tensor<256xf32>) -> tensor<?xf32>
// CHECK:         return %[[T0]] : tensor<?xf32>
func.func @torch.aten.matmul.2dx1d(%arg0: !torch.vtensor<[?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) : (tensor<256xf32>, tensor<256x?xf32>) -> tensor<?xf32>
// CHECK:         return %[[T0]] : tensor<?xf32>
func.func @torch.aten.matmul.1dx2d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
// CHECK:         return %[[T0]] : tensor<f32>
func.func @torch.aten.matmul.1dx1d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.proj(
// CHECK:         tensor.from_elements
// CHECK:         mhlo.dynamic_reshape
// CHECK:         mhlo.dot
// CHECK:         tensor.from_elements
// CHECK:         mhlo.dynamic_reshape
// CHECK-NOT:     mhlo.dot_general
func.func @torch.aten.matmul.proj(%arg0: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[?,?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,?,256],f32>
  return %1 : !torch.vtensor<[?,?,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mm.proj(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>) -> tensor<?x256xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot"(%[[ARG0]], %[[T0]]) : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         return %[[T1]] : tensor<?x256xf32>
func.func @torch.aten.mm.proj(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,256],f32>
  return %1 : !torch.vtensor<[?,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.dynamic_shape_cast(

func.func @torch.aten.matmul.dynamic_shape_cast(%arg0: !torch.vtensor<[2,?,?],f32>) -> !torch.vtensor<[2,256,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[2,?,?],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[2,256,256],f32>
  return %1 : !torch.vtensor<[2,256,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mm.dynamic_shape_cast(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>) -> tensor<2x256xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot"(%[[ARG0]], %[[T0]]) : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[T2:.*]] = tensor.cast %[[T1]] : tensor<?x256xf32> to tensor<2x256xf32>
// CHECK:         return %[[T2]] : tensor<2x256xf32>
func.func @torch.aten.mm.dynamic_shape_cast(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[2,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[2,256],f32>
  return %1 : !torch.vtensor<[2,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul.1dx2d.dynamic_shape_cast(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<1xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) : (tensor<?xf32>, tensor<256x?xf32>) -> tensor<?xf32>
// CHECK:         %[[T1:.*]] = tensor.cast %[[T0]] : tensor<?xf32> to tensor<1xf32>
// CHECK:         return %[[T1]] : tensor<1xf32>
func.func @torch.aten.matmul.1dx2d.dynamic_shape_cast(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[1],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

