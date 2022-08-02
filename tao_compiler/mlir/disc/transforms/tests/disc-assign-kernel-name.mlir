// RUN: disc-opt -disc-assign-kernel-name -split-input-file %s -o - | FileCheck %s

module @main attributes {gpu.container_module}  {
  gpu.module @kernel_module attributes {gpu.binary = "BLOB!"} {
    // CHECK: gpu.func @main_kLoop_divide__15_1_0
    gpu.func @the_kernel(%arg0: memref<?x?xf32>) kernel {
      gpu.return
    }
  }

  // CHECK: func.func @test_gpu_launch
  func.func @test_gpu_launch(%arg0: !disc_ral.context, %arg1: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    "lmhlo.fusion"() ({
      // CHECK: gpu.launch_func
      // CHECK-SAME: @kernel_module::@main_kLoop_divide__15_1_0
      gpu.launch_func  @kernel_module::@the_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg1 : memref<?x?xf32>)
      "lmhlo.terminator"() : () -> ()
    }) {disc.fusion.name = "main_kLoop_divide__15_1_0"} : () -> ()
    return
  }
}