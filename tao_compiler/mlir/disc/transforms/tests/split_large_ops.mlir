// RUN: disc-opt -split-large-ops=max-num-operands-per-op=2 %s | FileCheck %s

// CHECK-LABEL: func.func @test
func.func @test(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2, %arg3) { dimension = 0 : i64 } : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-COUNT-3: concatenate
  return %0 : tensor<?x?xf32>
}
