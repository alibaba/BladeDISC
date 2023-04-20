// RUN: disc-opt -disc-lmhlo-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: test_printf
module @main attributes {gpu.container_module}  {
  func.func @test_concat(%arg0: tensor<2x16xf32>, %arg1: tensor<2x16xf32>) -> tensor<4x32xf32> attributes {gpu.kernel} {
    %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x3xi32>, tensor<2x2xi32>) -> tensor<3x3xi32>
    return %0 : tensor<4x32xf32>
  }
}