// RUN: disc-opt -pass-pipeline='builtin.module(gpu.module(gpu.func(disc-side-effect-loop-invariant-code-motion)))' \
// RUN: -split-input-file %s | FileCheck %s


gpu.module @kernel_module {
  // CHECK-LABEL: @independent_load_in_loop
  gpu.func @independent_load_in_loop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xindex>) -> index {
    // CHECK: memref.load
    // CHECK: scf.for
    // CHECK-NOT: memref.load

    %c0 = arith.constant 0 : index
    %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %c0) -> (index) {
      %sn = arith.addi %si, %si : index
      %ld = memref.load %arg3[%c0] : memref<1xindex>
      %res = arith.addi %si, %ld : index
      scf.yield %res : index
    }
    gpu.return %result : index
  }

  // CHECK-LABEL: @dependent_load_in_loop
  gpu.func @dependent_load_in_loop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xindex>) -> index {
    // CHECK: scf.for
    // CHECK: memref.load

    %c0 = arith.constant 0 : index
    %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %c0) -> (index) {
      %sn = arith.addi %si, %si : index
      %ld = memref.load %arg3[%i0] : memref<1xindex>
      %res = arith.addi %si, %ld : index
      scf.yield %res : index
    }
    gpu.return %result : index
  }
}
