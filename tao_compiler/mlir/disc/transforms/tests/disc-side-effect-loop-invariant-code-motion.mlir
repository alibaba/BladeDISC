// RUN: disc-opt -pass-pipeline='gpu.module(gpu.func(disc-side-effect-loop-invariant-code-motion))' \
// RUN: -split-input-file %s | FileCheck %s


gpu.module @kernel_module {
  // CHECK-LABEL: @simple_loop
  gpu.func @simple_loop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xi32>) -> i32 {
    // CHECK: memref.load
    // CHECK: scf.for
    // CHECK-NOT: memref.load

    %s0 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (i32) {
      %sn = arith.addi %si, %si : i32
      %ld = memref.load %arg3[%c0] : memref<1xi32>
      %res = arith.addi %si, %ld : i32
      scf.yield %res : i32
    }
    gpu.return %result : i32
  }
}
