// RUN: disc-opt -disc-lhlo-rewriter -split-input-file %s -o - | FileCheck %s

module @main attributes {gpu.container_module}  {
  func.func @test_concat(%arg0: memref<2x16xf32, #gpu.address_space<global>>, %arg1: memref<2x16xf32, #gpu.address_space<global>>, %out : memref<4x16xf32, #gpu.address_space<global>>) -> memref<4x16xf32, #gpu.address_space<global>> attributes {gpu.kernel} {
    // CHECK: memref.alloc() : memref<3xi64>
    // CHECK: "disc_ral.get_pointer"(%arg0)
    // CHECK: memref.store %0, %alloc[%c0]
    // CHECK: "disc_ral.get_pointer"(%arg1)
    // CHECK: memref.store %1, %alloc[%c1]
    // CHECK: memref.alloc() : memref<3xi64>
    // CHECK: "lmhlo_disc.h2d"(%alloc, %alloc_0)
    // CHECK: "lmhlo_disc.concatenate"(%arg0, %arg1, %alloc_0, %arg2)
    "lmhlo.concatenate"(%arg0, %arg1, %out) { dimension = 0 : i64, disc.device = "gpu"} : (memref<2x16xf32, #gpu.address_space<global>>, memref<2x16xf32, #gpu.address_space<global>>, memref<4x16xf32, #gpu.address_space<global>>) -> ()
    return %out : memref<4x16xf32, #gpu.address_space<global>>
  }
}