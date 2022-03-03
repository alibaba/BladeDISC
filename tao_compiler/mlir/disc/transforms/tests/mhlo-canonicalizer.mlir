// RUN: disc-opt %s --disc-shape-simplifier --split-input-file | FileCheck %s

// CHECK-LABEL: func @main
func @main(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x2xf32> {
  // CHECK: mhlo.concatenate
  // CHECK-SAME: tensor<3x2xf32>
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @main
func @main(%arg0 : tensor<3x4xf32>) -> tensor<?x?xf32> {
  %begin = arith.constant dense<0> : tensor<2xi32>
  %end = arith.constant dense<[2,3]> : tensor<2xi32>
  %stride = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK: mhlo.slice
  %0 = "mhlo.real_dynamic_slice"(%arg0, %begin, %end, %stride) : (tensor<3x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
