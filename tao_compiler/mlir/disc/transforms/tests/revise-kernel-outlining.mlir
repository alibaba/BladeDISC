// RUN: disc-opt %s -disc-revise-gpu-kernel-outlining -split-input-file | FileCheck %s

// func @need_revise(%input1: memref<?x?x?xf32, "gpu">, %input2: memref<3xi32>, %input3: memref<3xi32>, %input4: memref<3xi32>, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
//   "lmhlo.fusion"() ( {
//     "lmhlo.real_dynamic_slice"(%input1, %input2, %input3, %input4, %out) : (memref<?x?x?xf32, "gpu">, memref<3xi32>, memref<3xi32>, memref<3xi32>, memref<?x?x?xf32, "gpu">) -> ()
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return %out : memref<?x?x?xf32, "gpu">
// }

#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  // CHECK-LABEL: @need_revision
  func @need_revision(%arg0: memref<?x?x?xf32, "gpu">, %arg1: memref<3xi32>, %arg2: memref<3xi32>, %arg3: memref<3xi32>, %arg4: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu"> {
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = memref.dim %arg4, %c0 : memref<?x?x?xf32, "gpu">
    %1 = memref.dim %arg4, %c1 : memref<?x?x?xf32, "gpu">
    %2 = muli %0, %1 : index
    %3 = memref.dim %arg4, %c2 : memref<?x?x?xf32, "gpu">
    %4 = muli %2, %3 : index
    %c0_0 = constant 0 : index
    %c256 = constant 256 : index
    %5 = muli %c1, %c256 : index
    %c1_1 = constant 1 : index
    %6 = affine.apply #map0(%4)[%c0, %5]
    %7 = affine.apply #map0(%5)[%c0_0, %c1]
    gpu.launch_func  @need_revision_kernel::@need_revision_kernel blocks in (%6, %c1_1, %c1_1) threads in (%7, %c1_1, %c1_1) args(%5 : index, %4 : index, %arg4 : memref<?x?x?xf32, "gpu">, %arg1 : memref<3xi32>, %arg3 : memref<3xi32>, %arg0 : memref<?x?x?xf32, "gpu">)
    return %arg4 : memref<?x?x?xf32, "gpu">
  }
  gpu.module @need_revision_kernel {
    gpu.func @need_revision_kernel(%arg0: index, %arg1: index, %arg2: memref<?x?x?xf32, "gpu">, %arg3: memref<3xi32>, %arg4: memref<3xi32>, %arg5: memref<?x?x?xf32, "gpu">) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.block_id"() {dimension = "y"} : () -> index
      %2 = "gpu.block_id"() {dimension = "z"} : () -> index
      %3 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %4 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %5 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %6 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      %7 = "gpu.grid_dim"() {dimension = "y"} : () -> index
      %8 = "gpu.grid_dim"() {dimension = "z"} : () -> index
      %9 = "gpu.block_dim"() {dimension = "x"} : () -> index
      %10 = "gpu.block_dim"() {dimension = "y"} : () -> index
      %11 = "gpu.block_dim"() {dimension = "z"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      %c0_0 = constant 0 : index
      %c2 = constant 2 : index
      %12 = affine.apply #map1(%0)[%arg0, %c0]
      %13 = affine.apply #map1(%3)[%c1, %c0_0]
      %14 = addi %13, %12 : index
      %true = constant true
      %15 = muli %13, %c1 : index
      %16 = addi %15, %12 : index
      %17 = cmpi ult, %16, %arg1 : index
      %18 = and %true, %17 : i1
      scf.if %18 {
        %19 = memref.dim %arg2, %c1 : memref<?x?x?xf32, "gpu">
        %20 = memref.dim %arg2, %c2 : memref<?x?x?xf32, "gpu">
        %21 = muli %20, %19 : index
        %22 = divi_unsigned %14, %21 : index
        %23 = remi_unsigned %14, %21 : index
        %24 = divi_unsigned %23, %20 : index
        %25 = remi_unsigned %23, %20 : index
        %26 = memref.load %arg3[%c0] : memref<3xi32>
        %27 = index_cast %26 : i32 to index
        %28 = memref.load %arg4[%c0] : memref<3xi32>
        %29 = index_cast %28 : i32 to index
        %30 = muli %22, %29 : index
        %31 = addi %30, %27 : index
        %32 = memref.load %arg3[%c1] : memref<3xi32>
        %33 = index_cast %32 : i32 to index
        %34 = memref.load %arg4[%c1] : memref<3xi32>
        %35 = index_cast %34 : i32 to index
        %36 = muli %24, %35 : index
        %37 = addi %36, %33 : index
        %38 = memref.load %arg3[%c2] : memref<3xi32>
        %39 = index_cast %38 : i32 to index
        %40 = memref.load %arg4[%c2] : memref<3xi32>
        %41 = index_cast %40 : i32 to index
        %42 = muli %25, %41 : index
        %43 = addi %42, %39 : index
        %44 = memref.load %arg5[%31, %37, %43] : memref<?x?x?xf32, "gpu">
        %45 = memref.dim %arg2, %c0 : memref<?x?x?xf32, "gpu">
        %46 = memref.dim %arg2, %c1 : memref<?x?x?xf32, "gpu">
        %47 = muli %45, %46 : index
        %48 = memref.dim %arg2, %c2 : memref<?x?x?xf32, "gpu">
        %49 = muli %47, %48 : index
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
