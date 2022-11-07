// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten._softmax_backward_data(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
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
// CHECK:         %[[T15:.*]] = chlo.broadcast_subtract %[[T1]], %[[T14]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
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
// CHECK:         %[[T1:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T1]]) applies mhlo.maximum across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T5:.*]] = tensor.from_elements %[[T3]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T6:.*]] = mhlo.dynamic_reshape %[[T4]], %[[T5]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:         %[[T7:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T6]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = mhlo.exponential %[[T7]] : tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = mhlo.reduce(%[[T8]] init: %[[T0]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T10:.*]] = tensor.dim %[[T8]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T11]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[T9]], %[[T12]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_divide %[[T8]], %[[T13]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T14]] : tensor<?x?xf32>
func.func @torch.aten._softmax(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int_1 = torch.constant.int -1
  %bool_false = torch.constant.bool false
  %result = torch.aten._softmax %arg0, %int_1, %bool_false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

