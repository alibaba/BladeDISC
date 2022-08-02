// RUN: disc-opt %s -disc-revise-gpu-kernel-outlining -split-input-file | FileCheck %s

// func.func @need_revise(%input1: memref<?x?x?xf32, "gpu">, %input2: memref<3xi32>, %input3: memref<3xi32>, %input4: memref<3xi32>, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
//   "lmhlo.fusion"() ({
//     "lmhlo.real_dynamic_slice"(%input1, %input2, %input3, %input4, %out) : (memref<?x?x?xf32, "gpu">, memref<3xi32>, memref<3xi32>, memref<3xi32>, memref<?x?x?xf32, "gpu">) -> ()
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return %out : memref<?x?x?xf32, "gpu">
// }

#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  // CHECK-LABEL: @need_revision
  func.func @need_revision(%arg0: memref<?x?x?xf32, "gpu">, %arg1: memref<3xi32>, %arg2: memref<3xi32>, %arg3: memref<3xi32>, %arg4: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu"> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg4, %c0 : memref<?x?x?xf32, "gpu">
    %1 = memref.dim %arg4, %c1 : memref<?x?x?xf32, "gpu">
    %2 = arith.muli %0, %1 : index
    %3 = memref.dim %arg4, %c2 : memref<?x?x?xf32, "gpu">
    %4 = arith.muli %2, %3 : index
    %c0_0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %5 = arith.muli %c1, %c256 : index
    %c1_1 = arith.constant 1 : index
    %6 = affine.apply #map0(%4)[%c0, %5]
    %7 = affine.apply #map0(%5)[%c0_0, %c1]
    gpu.launch_func  @need_revision_kernel::@need_revision_kernel blocks in (%6, %c1_1, %c1_1) threads in (%7, %c1_1, %c1_1) args(%5 : index, %4 : index, %arg4 : memref<?x?x?xf32, "gpu">, %arg1 : memref<3xi32>, %arg3 : memref<3xi32>, %arg0 : memref<?x?x?xf32, "gpu">)
    return %arg4 : memref<?x?x?xf32, "gpu">
  }
  gpu.module @need_revision_kernel {
    gpu.func @need_revision_kernel(%arg0: index, %arg1: index, %arg2: memref<?x?x?xf32, "gpu">, %arg3: memref<3xi32>, %arg4: memref<3xi32>, %arg5: memref<?x?x?xf32, "gpu">) kernel {
      %0 = gpu.block_id x
      %1 = gpu.block_id y
      %2 = gpu.block_id z
      %3 = gpu.thread_id x
      %4 = gpu.thread_id y
      %5 = gpu.thread_id z
      %6 = gpu.grid_dim x
      %7 = gpu.grid_dim y
      %8 = gpu.grid_dim z
      %9 = gpu.block_dim x
      %10 = gpu.block_dim y
      %11 = gpu.block_dim z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %12 = affine.apply #map1(%0)[%arg0, %c0]
      %13 = affine.apply #map1(%3)[%c1, %c0_0]
      %14 = arith.addi %13, %12 : index
      %true = arith.constant true
      %15 = arith.muli %13, %c1 : index
      %16 = arith.addi %15, %12 : index
      %17 = arith.cmpi ult, %16, %arg1 : index
      %18 = arith.andi %true, %17 : i1
      scf.if %18 {
        %19 = memref.dim %arg2, %c1 : memref<?x?x?xf32, "gpu">
        %20 = memref.dim %arg2, %c2 : memref<?x?x?xf32, "gpu">
        %21 = arith.muli %20, %19 : index
        %22 = arith.divui %14, %21 : index
        %23 = arith.remui %14, %21 : index
        %24 = arith.divui %23, %20 : index
        %25 = arith.remui %23, %20 : index
        %26 = memref.load %arg3[%c0] : memref<3xi32>
        %27 = arith.index_cast %26 : i32 to index
        %28 = memref.load %arg4[%c0] : memref<3xi32>
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.muli %22, %29 : index
        %31 = arith.addi %30, %27 : index
        %32 = memref.load %arg3[%c1] : memref<3xi32>
        %33 = arith.index_cast %32 : i32 to index
        %34 = memref.load %arg4[%c1] : memref<3xi32>
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.muli %24, %35 : index
        %37 = arith.addi %36, %33 : index
        %38 = memref.load %arg3[%c2] : memref<3xi32>
        %39 = arith.index_cast %38 : i32 to index
        %40 = memref.load %arg4[%c2] : memref<3xi32>
        %41 = arith.index_cast %40 : i32 to index
        %42 = arith.muli %25, %41 : index
        %43 = arith.addi %42, %39 : index
        %44 = memref.load %arg5[%31, %37, %43] : memref<?x?x?xf32, "gpu">
        %45 = memref.dim %arg2, %c0 : memref<?x?x?xf32, "gpu">
        %46 = memref.dim %arg2, %c1 : memref<?x?x?xf32, "gpu">
        %47 = arith.muli %45, %46 : index
        %48 = memref.dim %arg2, %c2 : memref<?x?x?xf32, "gpu">
        %49 = arith.muli %47, %48 : index
        %50 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%49], strides: [%c1] : memref<?x?x?xf32, "gpu"> to memref<?xf32, "gpu">
        memref.store %44, %50[%14] : memref<?xf32, "gpu">
      }
      gpu.return
    }
  }
}
// CHECK:    gpu.launch_func  @need_revision_kernel::@need_revision_kernel
// CHECK:  gpu.module @need_revision_kernel {
// CHECK:    gpu.func @need_revision_kernel
// CHECK-NOT: memref<3xi32>
