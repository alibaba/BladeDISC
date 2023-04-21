// RUN: disc-opt -disc-lhlo-rewriter -split-input-file %s -o - | FileCheck %s

module @main attributes {gpu.container_module}  {
  func.func @test_concat(%arg0: memref<2x16xf32>, %arg1: memref<2x16xf32>, %out : memref<4x16xf32>) -> memref<4x16xf32> attributes {gpu.kernel} {
    // CHECK:     lmhlo_disc.concatenate
    "lmhlo.concatenate"(%arg0, %arg1, %out) { dimension = 0 : i64 } : (memref<2x16xf32>, memref<2x16xf32>, memref<4x16xf32>) -> ()
    return %out : memref<4x16xf32>
  }
}