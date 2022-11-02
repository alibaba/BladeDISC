// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.native_dropout.train(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi1>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T6:.*]] = arith.trunci %[[T3]] : i64 to i32
// CHECK:         %[[T7:.*]] = arith.trunci %[[T5]] : i64 to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T6]], %[[T7]] : tensor<2xi32>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T8]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = tensor.dim %[[T9]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T12:.*]] = tensor.dim %[[T9]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T13:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T11]], %[[T13]] : tensor<2xi32>
// CHECK:         %[[T15:.*]] = "mhlo.rng"(%[[T1]], %[[T0]], %[[T14]]) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T16:.*]] = chlo.broadcast_compare %[[T15]], %[[T1]] {compare_type = #chlo<comparison_type FLOAT>, comparison_direction = #chlo<comparison_direction LT>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:         %[[T17:.*]] = mhlo.convert %[[T16]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T17]], %[[ARG0]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T19:.*]] = chlo.broadcast_multiply %[[T18]], %[[T1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T20:.*]] = chlo.broadcast_compare %[[T17]], %[[T0]] {compare_type = #chlo<comparison_type FLOAT>, comparison_direction = #chlo<comparison_direction GE>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:         return %[[T19]], %[[T20]] : tensor<?x?xf32>, tensor<?x?xi1>
func.func @torch.aten.native_dropout.train(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>) {
  %float1 = torch.constant.float 1.000000e+00
  %bool_true = torch.constant.bool true
  %result0, %result1 = torch.aten.native_dropout %arg0, %float1, %bool_true: !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
  return %result0, %result1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_dropout_backward(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xi1>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.convert %[[ARG1]] : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[T1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = chlo.broadcast_multiply %[[T2]], %[[T0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         return %[[T3]] : tensor<?x?xf32>
func.func @torch.aten.native_dropout_backward(%arg0: !torch.vtensor<[?,?],f32>, %arg1 : !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],f32>{
  %float_1 = torch.constant.float 1.000000
  %result = torch.aten.native_dropout_backward %arg0, %arg1, %float_1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

