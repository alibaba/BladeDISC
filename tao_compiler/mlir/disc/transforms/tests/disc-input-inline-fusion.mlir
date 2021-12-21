// RUN: disc-opt %s -disc-input-inline-fusion | FileCheck %s

// CHECK-LABEL: @should_not_reuse_val_cache
func @should_not_reuse_val_cache(%arg0: memref<?xf32>, %arg1: memref<3xi32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>, %arg7: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  %c2 = constant 2 : index
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK-NOT: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg4) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%arg2, %arg4, %arg5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    %0 = memref.dim %arg6, %c0 : memref<?x?x?xf32>
    %1 = memref.dim %arg6, %c1 : memref<?x?x?xf32>
    %2 = muli %0, %1 : index
    %3 = memref.dim %arg6, %c2 : memref<?x?x?xf32>
    %4 = muli %2, %3 : index
    // CHECK: scf.parallel
    scf.parallel (%arg8) = (%c0) to (%4) step (%c1) {
      %5 = memref.dim %arg3, %c1 : memref<?x?x?xf32>
      %6 = memref.dim %arg3, %c2 : memref<?x?x?xf32>
      %7 = muli %6, %5 : index
      %8 = divi_unsigned %arg8, %7 : index
      %9 = remi_unsigned %arg8, %7 : index
      %10 = divi_unsigned %9, %6 : index
      %11 = remi_unsigned %9, %6 : index
      %12 = memref.load %arg3[%8, %10, %11] : memref<?x?x?xf32>
      %pred = cmpi eq, %7, %11 : index 
      // CHECK: scf.if
      scf.if %pred {
        // CHECK: addf
        // CHECK: divf 
        %13 = memref.load %arg5[%8, %10, %11] : memref<?x?x?xf32>
        %14 = divf %12, %13 : f32
        %15 = memref.reinterpret_cast %arg7 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
        memref.store %14, %15[%arg8] : memref<?xf32>
      }
      // CHECK: addf
      // CHECK: mulf 
      %16 = memref.load %arg5[%8, %10, %11] : memref<?x?x?xf32>
      %17 = mulf %12, %16 : f32
      %18 = memref.reinterpret_cast %arg6 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %17, %18[%arg8] : memref<?xf32>
      scf.yield
    }
    // CHECK: lmhlo.terminator
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  return %arg6, %arg7 : memref<?x?x?xf32>, memref<?x?x?xf32>
}
