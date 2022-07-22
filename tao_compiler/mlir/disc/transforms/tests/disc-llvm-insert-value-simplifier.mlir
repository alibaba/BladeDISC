// RUN: disc-opt %s -disc-llvm-insert-value-simplifier | FileCheck %s

llvm.func @__nv_fabsf(f32) -> f32
// CHECK-LABEL: @abs2D
llvm.func @abs2D(%arg0: i32, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !llvm.ptr<f32>, %arg9: !llvm.ptr<f32>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {gpu.kernel, nvvm.kernel} {
  // CHECK-NOT: llvm.insertvalue
  // CHECK-NOT: llvm.extractvalue
  %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %9 = llvm.insertvalue %arg8, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %10 = llvm.insertvalue %arg9, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %11 = llvm.insertvalue %arg10, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %12 = llvm.insertvalue %arg11, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %13 = llvm.insertvalue %arg13, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %14 = llvm.insertvalue %arg12, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %15 = llvm.insertvalue %arg14, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %16 = llvm.mlir.constant(10000 : index) : i32
  %17 = nvvm.read.ptx.sreg.ctaid.x : i32
  %18 = nvvm.read.ptx.sreg.tid.x : i32
  llvm.br ^bb1
^bb1:  // pred: ^bb0
  %19 = llvm.mul %17, %arg0  : i32
  %20 = llvm.add %18, %19  : i32
  %21 = llvm.add %18, %19  : i32
  %22 = llvm.icmp "ult" %21, %16 : i32
  llvm.cond_br %22, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %23 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %24 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %25 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %26 = llvm.insertvalue %24, %23[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %27 = llvm.insertvalue %25, %26[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %28 = llvm.mlir.constant(0 : index) : i32
  %29 = llvm.insertvalue %28, %27[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %30 = llvm.mlir.constant(10000 : index) : i32
  %31 = llvm.insertvalue %30, %29[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %32 = llvm.mlir.constant(1 : index) : i32
  %33 = llvm.insertvalue %32, %31[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %34 = llvm.extractvalue %33[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %35 = llvm.getelementptr %34[%20] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
  %36 = llvm.load %35 : !llvm.ptr<f32>
  %37 = llvm.call @__nv_fabsf(%36) : (f32) -> f32
  %38 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %39 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %40 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %41 = llvm.insertvalue %39, %38[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %42 = llvm.insertvalue %40, %41[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %43 = llvm.mlir.constant(0 : index) : i32
  %44 = llvm.insertvalue %43, %42[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %45 = llvm.mlir.constant(10000 : index) : i32
  %46 = llvm.insertvalue %45, %44[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %47 = llvm.mlir.constant(1 : index) : i32
  %48 = llvm.insertvalue %47, %46[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %49 = llvm.extractvalue %48[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
  %50 = llvm.getelementptr %49[%20] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
  llvm.store %37, %50 : !llvm.ptr<f32>
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  llvm.return
}