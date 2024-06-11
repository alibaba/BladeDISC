// RUN: disc-opt -split-input-file --disc-shape-propagate %s | FileCheck %s

// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101xi64>, %arg1: tensor<4x101xi64>) -> tensor<4x101xi1> attributes{tf.entry_function = {input_dynamic_dims = "0:1|1:1"}}{
  // CHECK: %0 = mhlo.compare  LT, %arg0, %arg1 : (tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<4x?xi1>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<4x101xi64>, tensor<4x101xi64>) -> tensor<4x101xi1>
  // CHECK: return %0 : tensor<4x?xi1>
  return %0 : tensor<4x101xi1>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101xi64>) -> tensor<4x101xi1> attributes{tf.entry_function = {input_dynamic_dims = "0:1"}}{
  // CHECK: %1 = shape.shape_of %arg0 : tensor<4x?xi64> -> tensor<2xindex>
  // CHECK: %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>, tensor<2xindex>) -> tensor<4x?xi64>
  %0 = mhlo.constant dense<0> : tensor<4x101xi64>
  %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<4x101xi64>, tensor<4x101xi64>) -> tensor<4x101xi1>
  return %1 : tensor<4x101xi1>
}