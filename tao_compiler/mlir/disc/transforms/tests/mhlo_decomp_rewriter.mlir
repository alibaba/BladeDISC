// RUN: disc-opt -disc-mhlo-decomp-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL:  func.func @batch_norm_inference(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x128x?x?xf32>, %[[ARG1:.*]]: tensor<128xf32>, %[[ARG2:.*]]: tensor<128xf32>, %[[ARG3:.*]]: tensor<128xf32>, %[[ARG4:.*]]: tensor<128xf32>) -> tensor<?x128x?x?xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[CST:.*]] = arith.constant dense<128> : tensor<1xindex>
// CHECK:         %[[C128:.*]] = arith.constant 128 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[CST]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<128xf32>
// CHECK:         %[[T2:.*]] = mhlo.add %[[ARG4]], %[[T1]] : tensor<128xf32>
// CHECK:         %[[T3:.*]] = mhlo.rsqrt %[[T2]] : tensor<128xf32>
// CHECK:         %[[T4:.*]] = mhlo.multiply %[[T3]], %[[ARG1]] : tensor<128xf32>
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T5]], %[[C128]], %[[T6]], %[[T7]] : tensor<4xindex>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T4]], %[[T8]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>, tensor<4xindex>) -> tensor<?x128x?x?xf32>
// CHECK:         %[[T10:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T11:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T12:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T13:.*]] = tensor.from_elements %[[T10]], %[[C128]], %[[T11]], %[[T12]] : tensor<4xindex>
// CHECK:         %[[T14:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG3]], %[[T13]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>, tensor<4xindex>) -> tensor<?x128x?x?xf32>
// CHECK:         %[[T15:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T16:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T17:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T18:.*]] = tensor.from_elements %[[T15]], %[[C128]], %[[T16]], %[[T17]] : tensor<4xindex>
// CHECK:         %[[T19:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[T18]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>, tensor<4xindex>) -> tensor<?x128x?x?xf32>
// CHECK:         %[[T20:.*]] = mhlo.subtract %[[ARG0]], %[[T14]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T21:.*]] = mhlo.multiply %[[T20]], %[[T9]] : tensor<?x128x?x?xf32>
// CHECK:         %[[T22:.*]] = mhlo.add %[[T21]], %[[T19]] : tensor<?x128x?x?xf32>
// CHECK:         return %[[T22]] : tensor<?x128x?x?xf32>
func.func @batch_norm_inference(%arg0: tensor<?x128x?x?xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<?x128x?x?xf32> {
  %0 = "mhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
  return %0: tensor<?x128x?x?xf32>
}
