// RUN: disc-opt %s -disc-function-dead-argument-elimination | FileCheck %s

// CHECK-LABEL: @main_kernel
gpu.module @main_kernel {
  llvm.func @__nv_fabsf(f32) -> f32
  // CHECK:      llvm.func @abs2D(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: !llvm.ptr<f32>, %[[ARG2:.*]]: !llvm.ptr<f32>)
  // CHECK-SAME: attributes {disc.elimargs = [1 : index, 3 : index, 4 : index, 5 : index, 6 : index, 7 : index, 8 : index, 10 : index, 11 : index, 12 : index, 13 : index, 14 : index]
  llvm.func @abs2D(%arg0: i32, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !llvm.ptr<f32>, %arg9: !llvm.ptr<f32>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.constant(10000 : index) : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mul %1, %arg0  : i32
    %4 = llvm.add %2, %3  : i32
    %5 = llvm.add %2, %3  : i32
    %6 = llvm.icmp "ult" %5, %0 : i32
    llvm.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %7 = llvm.getelementptr %arg2[%4] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    %8 = llvm.load %7 : !llvm.ptr<f32>
    %9 = llvm.call @__nv_fabsf(%8) : (f32) -> f32
    %10 = llvm.getelementptr %arg9[%4] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    llvm.store %9, %10 : !llvm.ptr<f32>
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    llvm.return
  }
}