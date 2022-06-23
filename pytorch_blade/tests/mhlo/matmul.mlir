// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.mm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x3xf32>) -> tensor<2x3xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
// CHECK:         return %[[T0]] : tensor<2x3xf32>
func @torch.aten.mm(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.bmm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<10x3x4xf32>, %[[ARG1:.*]]: tensor<10x4x5xf32>) -> tensor<10x3x5xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<10x3x4xf32>, tensor<10x4x5xf32>) -> tensor<10x3x5xf32>
// CHECK:         return %[[T0]] : tensor<10x3x5xf32>
func @torch.aten.bmm(%arg0: !torch.vtensor<[10,3,4],f32>, %arg1: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,3,4],f32>, !torch.vtensor<[10,4,5],f32> -> !torch.vtensor<[10,3,5],f32>
  return %0 : !torch.vtensor<[10,3,5],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.bmm.dyn(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x?x256xf32>, %[[ARG1:.*]]: tensor<4x256x?xf32>) -> tensor<4x?x?xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x?x256xf32>, tensor<4x256x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         return %[[T0]] : tensor<4x?x?xf32>
func @torch.aten.bmm.dyn(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[4,256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.dyn(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x?x256xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<4x?x?xf32> {
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<4x?x256xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = arith.muli %[[T1]], %[[C4_I32]] : i32
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]], %[[C256_I32]] : tensor<2xi32>
// CHECK:         %[[T4:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[T3]]) : (tensor<4x?x256xf32>, tensor<2xi32>) -> tensor<?x256xf32>
// CHECK:         %[[T5:.*]] = "mhlo.dot_general"(%[[T4]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = tensor.dim %[[T5]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[C4_I32]], %[[T1]], %[[T7]] : tensor<3xi32>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_reshape"(%[[T5]], %[[T8]]) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T10:.*]] = mhlo.convert(%[[T9]]) : (tensor<?x?x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         return %[[T10]] : tensor<4x?x?xf32>
func @torch.aten.matmul.dyn(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256x120xf32>, %[[ARG1:.*]]: tensor<4x120x256xf32>) -> tensor<4x256x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.broadcast_in_dim"(%[[ARG0]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<256x120xf32>) -> tensor<4x256x120xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[T0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x256x120xf32>, tensor<4x120x256xf32>) -> tensor<4x256x256xf32>
// CHECK:         return %[[T1]] : tensor<4x256x256xf32>
func @torch.aten.matmul(%arg0: !torch.vtensor<[256,120],f32>, %arg1: !torch.vtensor<[4,120,256],f32>) -> !torch.vtensor<[4,256,256],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256,120],f32>, !torch.vtensor<[4,120,256],f32> -> !torch.vtensor<[4,256,256],f32>
  return %0 : !torch.vtensor<[4,256,256],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.3dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1x?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<1x?xf32> {
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<256xf32>) -> tensor<256x1xf32>
// CHECK:         %[[T1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<1x?x256xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i32
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]], %[[C256_I32]] : tensor<2xi32>
// CHECK:         %[[T4:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[T3]]) : (tensor<1x?x256xf32>, tensor<2xi32>) -> tensor<?x256xf32>
// CHECK:         %[[T5:.*]] = "mhlo.dot_general"(%[[T4]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[C1_I32]], %[[T2]], %[[C1_I32]] : tensor<3xi32>
// CHECK:         %[[T7:.*]] = "mhlo.dynamic_reshape"(%[[T5]], %[[T6]]) : (tensor<?x1xf32>, tensor<3xi32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T8:.*]] = tensor.dim %[[T7]], %[[C0]] : tensor<?x?x1xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = tensor.dim %[[T7]], %[[C1]] : tensor<?x?x1xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T9]], %[[T11]] : tensor<2xi32>
// CHECK:         %[[T13:.*]] = "mhlo.dynamic_reshape"(%[[T5]], %[[T12]]) : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T14:.*]] = mhlo.convert(%[[T13]]) : (tensor<?x?xf32>) -> tensor<1x?xf32>
// CHECK:         return %[[T14]] : tensor<1x?xf32>
func @torch.aten.matmul.3dx1d(%arg0: !torch.vtensor<[1,?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[1,?],f32>
  return %0 : !torch.vtensor<[1,?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.1dx3d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<?x256x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<256xf32>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x256x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i32
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]], %[[C1_I32]], %[[C256_I32]] : tensor<3xi32>
// CHECK:         %[[T4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[T3]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x256xf32>, tensor<3xi32>) -> tensor<?x1x256xf32>
// CHECK:         %[[T5:.*]] = "mhlo.dot_general"(%[[T4]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x1x256xf32>, tensor<?x256x?xf32>) -> tensor<?x1x?xf32>
// CHECK:         %[[T6:.*]] = tensor.dim %[[T5]], %[[C0]] : tensor<?x1x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[T5]], %[[C2]] : tensor<?x1x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T7]], %[[T9]] : tensor<2xi32>
// CHECK:         %[[T11:.*]] = "mhlo.dynamic_reshape"(%[[T5]], %[[T10]]) : (tensor<?x1x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         return %[[T11]] : tensor<?x?xf32>
func @torch.aten.matmul.1dx3d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[?,256,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<?xf32> {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<256xf32>) -> tensor<256x1xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x1xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]] : tensor<1xi32>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T1]], %[[T4]]) : (tensor<?x1xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK:         return %[[T5]] : tensor<?xf32>
func @torch.aten.matmul.2dx1d(%arg0: !torch.vtensor<[?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<256xf32>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[T0]], %[[ARG1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x?xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<1x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]] : tensor<1xi32>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T1]], %[[T4]]) : (tensor<1x?xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK:         return %[[T5]] : tensor<?xf32>
func @torch.aten.matmul.1dx2d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<256xf32>, %[[ARG1:.*]]: tensor<256xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<256xf32>) -> tensor<1x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<256xf32>) -> tensor<256x1xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:         %[[T3:.*]] = "mhlo.reshape"(%[[T2]]) : (tensor<1x1xf32>) -> tensor<f32>
// CHECK:         return %[[T3]] : tensor<f32>
func @torch.aten.matmul.1dx1d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.matmul.proj(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x256xf32>) -> tensor<?x?x256xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C256_I32:.*]] = arith.constant 256 : i32
// CHECK:         %[[T1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x256xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i32
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x256xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = arith.muli %[[T2]], %[[T4]] : i32
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T5]], %[[C256_I32]] : tensor<2xi32>
// CHECK:         %[[T7:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[T6]]) : (tensor<?x?x256xf32>, tensor<2xi32>) -> tensor<?x256xf32>
// CHECK:         %[[T8:.*]] = "mhlo.dot_general"(%[[T7]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T2]], %[[T4]], %[[C256_I32]] : tensor<3xi32>
// CHECK:         %[[T10:.*]] = "mhlo.dynamic_reshape"(%[[T8]], %[[T9]]) : (tensor<?x256xf32>, tensor<3xi32>) -> tensor<?x?x256xf32>
// CHECK:         return %[[T10]] : tensor<?x?x256xf32>
func @torch.aten.matmul.proj(%arg0: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[?,?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,?,256],f32>
  return %1 : !torch.vtensor<[?,?,256],f32>
}


// -----
// CHECK-LABEL:  func @torch.aten.mm.proj(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x256xf32>) -> tensor<?x256xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T1:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[T0]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         return %[[T1]] : tensor<?x256xf32>
func @torch.aten.mm.proj(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,256],f32>
  return %1 : !torch.vtensor<[?,256],f32>
}

