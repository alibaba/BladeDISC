
// RUN: disc-opt -split-input-file -disc-hlo-legalize-to-lhlo -hlo-legalize-to-lhlo -disc-optimization-barrier-expand %s -o - | FileCheck %s


// CHECK-LABEL: @optimization_barrier_expand
func.func @optimization_barrier_expand(%arg0 : tensor<1x2048x4096xf32>, %arg1: tensor<1x2048x4096xf32>) -> tensor<2048x4096xf16> {
  // CHECK: %alloc = memref.alloc() : memref<1x2048x4096xf32>
  // CHECK: "lmhlo.add"(%arg0, %arg1, %alloc) : (memref<1x2048x4096xf32>, memref<1x2048x4096xf32>, memref<1x2048x4096xf32>) -> ()
  %1 = "mhlo.add"(%arg0, %arg1): (tensor<1x2048x4096xf32>, tensor<1x2048x4096xf32>) -> tensor<1x2048x4096xf32>
  // CHECK: %alloc_0 = memref.alloc() : memref<1x2048x4096xf32>
  // CHECK: "lmhlo.add"(%arg0, %arg1, %alloc_0) : (memref<1x2048x4096xf32>, memref<1x2048x4096xf32>, memref<1x2048x4096xf32>) -> ()
  %2 = "mhlo.add"(%arg0, %arg1): (tensor<1x2048x4096xf32>, tensor<1x2048x4096xf32>) -> tensor<1x2048x4096xf32>
  // CHECK: %alloc_1 = memref.alloc() : memref<1x2048x4096xf16>
  // CHECK: "lmhlo.convert"(%alloc_0, %alloc_1) : (memref<1x2048x4096xf32>, memref<1x2048x4096xf16>) -> ()
  %3:2 = "mhlo.optimization_barrier"(%1, %2): (tensor<1x2048x4096xf32>, tensor<1x2048x4096xf32>) -> (tensor<1x2048x4096xf32>, tensor<1x2048x4096xf32>)
  %4 = "mhlo.convert"(%3#1): (tensor<1x2048x4096xf32>) -> tensor<1x2048x4096xf16>
  // CHECK: %alloc_2 = memref.alloc() : memref<2048x4096xf16>
  // CHECK: "lmhlo.reshape"(%alloc_1, %alloc_2) : (memref<1x2048x4096xf16>, memref<2048x4096xf16>) -> ()
  %5 = "mhlo.reshape"(%4) : (tensor<1x2048x4096xf16>) -> tensor<2048x4096xf16>
  // CHECK: return %alloc_2 : memref<2048x4096xf16>
  return %5: tensor<2048x4096xf16>
}