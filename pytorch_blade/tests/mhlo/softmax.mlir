// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten._softmax_backward_data(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[T1]] init: %[[T0]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T3:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = tensor.from_elements %[[T4]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T6:.*]] = mhlo.dynamic_reshape %[[T2]], %[[T5]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i64
// CHECK:         %[[T10:.*]] = arith.trunci %[[T7]] : i64 to i32
// CHECK:         %[[T11:.*]] = arith.trunci %[[T9]] : i64 to i32
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T10]], %[[T11]] : tensor<2xi32>
// CHECK:         %[[T13:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T6]], %[[T12]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_multiply %[[ARG1]], %[[T13]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_subtract %[[T1]], %[[T1]]4 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T15]] : tensor<?x?xf32>
func.func @torch.aten._softmax_backward_data(%arg0 : !torch.vtensor<[?,?],f32>, %arg1 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32>{
  %int_1 = torch.constant.int 1
  %int_6 = torch.constant.int 6
  %result = torch.aten._softmax_backward_data %arg0, %arg1, %int_1, %int_6 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten._softmax(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = tensor.from_elements %[[T4]], %[[T6]] : tensor<2xi32>
// CHECK:         %[[T8:.*]] = "mhlo.slice"(%[[T7]]) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_iota"(%[[T8]]) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
// CHECK:         %[[T10:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T9]], %[[T7]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
// CHECK:         %[[T11:.*]]:2 = mhlo.reduce(%[[ARG0]] init: %[[T2]]), (%[[T10]] init: %[[T1]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i32>, %[[ARG4:.*]]: tensor<i32>)  {
// CHECK:         %[[T22:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T23:.*]] = "mhlo.select"(%[[T22]], %[[ARG1]], %[[ARG3]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T24:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T25:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i32>
// CHECK:         %[[T26:.*]] = "mhlo.select"(%[[T22]], %[[ARG2]], %[[ARG4]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:         %[[T27:.*]] = "mhlo.select"(%[[T24]], %[[T25]], %[[T26]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:         mhlo.return %[[T23]], %[[T27]] : tensor<f32>, tensor<i32>
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T4]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[T11]]#0, %[[T12]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T13]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T15:.*]] = mhlo.exponential %[[T14]] : tensor<?x?xf32>
// CHECK:         %[[T16:.*]] = mhlo.reduce(%[[T15]] init: %[[T0]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T17:.*]] = tensor.dim %[[T15]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[T17]] : index to i32
// CHECK:         %[[T19:.*]] = tensor.from_elements %[[T18]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T20:.*]] = mhlo.dynamic_reshape %[[T16]], %[[T19]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:         %[[T21:.*]] = chlo.broadcast_divide %[[T15]], %[[T20]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T21]] : tensor<?x?xf32>
func.func @torch.aten._softmax(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int_1 = torch.constant.int -1
  %bool_false = torch.constant.bool false
  %result = torch.aten._softmax %arg0, %int_1, %bool_false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

