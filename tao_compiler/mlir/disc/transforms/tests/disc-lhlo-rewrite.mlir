// RUN: disc-opt -disc-lhlo-rewriter -split-input-file %s -o - | FileCheck %s

module @main attributes {gpu.container_module}  {
  func.func @test_concat(%arg0: memref<2x16xf32, "gpu">, %arg1: memref<2x16xf32, "gpu">, %out : memref<4x16xf32, "gpu">) -> memref<4x16xf32, "gpu"> attributes {gpu.kernel} {
    // CHECK:     lmhlo_disc.concatenate
    "lmhlo.concatenate"(%arg0, %arg1, %out) { dimension = 0 : i64 } : (memref<2x16xf32, "gpu">, memref<2x16xf32, "gpu">, memref<4x16xf32, "gpu">) -> ()
    return %out : memref<4x16xf32, "gpu">
  }
}