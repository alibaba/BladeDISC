// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s


// CHECK-LABEL:     func.func @torch.aten._softmax_backward_data(
// CHECK-SAME:                    %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:             %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:             %[[CST_1:.*]] = arith.constant 1 : index
// CHECK:             %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:             %[[CST_1_I32:.*]] = arith.constant 1 : i32
// CHECK:             %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:             %[[T2:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:             %[[T3:.*]] = mhlo.reduce(%[[T2]] init: %[[T1]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:             %[[T4:.*]] = tensor.dim %[[T2]], %[[CST_0]] : tensor<?x?xf32>
// CHECK:             %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:             %[[T6:.*]] = tensor.from_elements %[[T5]], %[[CST_1_I32]] : tensor<2xi32>
// CHECK:             %[[T7:.*]] = "mhlo.dynamic_reshape"(%[[T3]], %[[T6]]) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:             %[[T8:.*]] = tensor.dim %[[T2]], %[[CST_1]] : tensor<?x?xf32>
// CHECK:             %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:             %[[T10:.*]] = tensor.from_elements %[[T5]], %[[T9]] : tensor<2xi32>
// CHECK:             %[[T11:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T7]], %[[T10]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:             %[[T12:.*]] = chlo.broadcast_multiply %[[ARG1]], %[[T11]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:             %[[T13:.*]] = chlo.broadcast_multiply %[[T12]], %[[T0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:             %[[T14:.*]] = chlo.broadcast_subtract %[[T2]], %[[T13]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:             return %[[T14]] : tensor<?x?xf32>
// CHECK            }
func.func @torch.aten._softmax_backward_data(%arg0 : !torch.vtensor<[?,?],f32>, %arg1 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32>{
  %int_1 = torch.constant.int 1
  %int_6 = torch.constant.int 6
  %result = torch.aten._softmax_backward_data %arg0, %arg1, %int_1, %int_6 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:     func.func @torch.aten._softmax(
// CHECK-SAME:                      %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:             %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:             %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:             %[[CST_1_I32:.*]] = arith.constant 1 : i32
// CHECK:             %[[T2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:             %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:             %[[T3:.*]] = tensor.dim %[[ARG0]], %[[CST_0]] : tensor<?x?xf32>
// CHECK:             %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:             %[[T5:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.maximum across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:             %[[T6:.*]] = tensor.from_elements %[[T4]], %[[CST_1_I32]] : tensor<2xi32>
// CHECK:             %[[T7:.*]] = "mhlo.dynamic_reshape"(%[[T5]], %[[T6]]) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:             %[[T8:.*]] = chlo.broadcast_multiply %[[T7]], %[[T0]] : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
// CHECK:             %[[T9:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T8]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:             %[[T10:.*]] = mhlo.exponential %[[T9]] : tensor<?x?xf32>
// CHECK:             %[[T11:.*]] = mhlo.reduce(%[[T10]] init: %[[T1]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:             %[[T12:.*]] = tensor.dim %[[T10]], %[[CST_0]] : tensor<?x?xf32>
// CHECK:             %[[T13:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:             %[[T14:.*]] = tensor.from_elements %[[T13]], %[[CST_1_I32]] : tensor<2xi32>
// CHECK:             %[[T15:.*]] = "mhlo.dynamic_reshape"(%[[T11]], %[[T14]]) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
// CHECK:             %[[T16:.*]] = chlo.broadcast_divide %[[T10]], %[[T15]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
// CHECK:             return %[[T16]] : tensor<?x?xf32>
// CHECK:           }
func.func @torch.aten._softmax(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int_1 = torch.constant.int -1
  %bool_false = torch.constant.bool false
  %result = torch.aten._softmax %arg0, %int_1, %bool_false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}