// RUN: disc-opt %s -disc-input-inline-fusion | FileCheck %s

// CHECK-LABEL: @should_not_reuse_val_cache
func.func @should_not_reuse_val_cache(%arg0: memref<?xf32>, %arg1: memref<3xi32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>, %arg7: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: "lmhlo.fusion"() ({
  "lmhlo.fusion"() ({
    // CHECK-NOT: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg4) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%arg2, %arg4, %arg5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    %0 = memref.dim %arg6, %c0 : memref<?x?x?xf32>
    %1 = memref.dim %arg6, %c1 : memref<?x?x?xf32>
    %2 = arith.muli %0, %1 : index
    %3 = memref.dim %arg6, %c2 : memref<?x?x?xf32>
    %4 = arith.muli %2, %3 : index
    // CHECK: scf.parallel
    scf.parallel (%arg8) = (%c0) to (%4) step (%c1) {
      %5 = memref.dim %arg3, %c1 : memref<?x?x?xf32>
      %6 = memref.dim %arg3, %c2 : memref<?x?x?xf32>
      %7 = arith.muli %6, %5 : index
      %8 = arith.divui %arg8, %7 : index
      %9 = arith.remui %arg8, %7 : index
      %10 = arith.divui %9, %6 : index
      %11 = arith.remui %9, %6 : index
      %12 = memref.load %arg3[%8, %10, %11] : memref<?x?x?xf32>
      %pred = arith.cmpi eq, %7, %11 : index
      // CHECK: scf.if
      scf.if %pred {
        // CHECK: addf
        // CHECK: arith.divf
        %13 = memref.load %arg5[%8, %10, %11] : memref<?x?x?xf32>
        %14 = arith.divf %12, %13 : f32
        %15 = memref.reinterpret_cast %arg7 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
        memref.store %14, %15[%arg8] : memref<?xf32>
      }
      // CHECK: addf
      // CHECK: arith.mulf
      %16 = memref.load %arg5[%8, %10, %11] : memref<?x?x?xf32>
      %17 = arith.mulf %12, %16 : f32
      %18 = memref.reinterpret_cast %arg6 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %17, %18[%arg8] : memref<?xf32>
      scf.yield
    }
    // CHECK: lmhlo.terminator
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  return %arg6, %arg7 : memref<?x?x?xf32>, memref<?x?x?xf32>
}
