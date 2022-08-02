// RUN: disc-opt -disc-shape-simplifier -split-input-file %s | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %cst = arith.constant dense<[2, 3, 4]> : tensor<3xindex>
  // CHECK: mhlo.reshape
  // CHECK-SAME: -> tensor<2x3x4xf32>
  %0 = "mhlo.dynamic_reshape"(%arg0, %cst) : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: index) -> tensor<?x?x?xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape = tensor.from_elements %arg1, %c3, %c4 : tensor<3xindex>
  // CHECK: mhlo.dynamic_reshape
  // CHECK-SAME: -> tensor<?x3x4xf32>
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
