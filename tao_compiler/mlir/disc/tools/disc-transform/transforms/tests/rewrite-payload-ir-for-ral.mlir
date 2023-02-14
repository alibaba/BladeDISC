// RUN: disc-opt --disc-rewrite-payload-ir-for-ral -split-input-file %s | FileCheck %s

#map = affine_map<()[s0] -> (s0 ceildiv 6)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map2 = affine_map<(d0)[s0] -> (d0 * -6 + s0, 6)>
#map3 = affine_map<(d0)[s0] -> (d0 * -16 + s0, 16)>
#map4 = affine_map<(d0) -> (d0 * 6)>
#map5 = affine_map<(d0) -> (d0 * 16)>
module {
  // CHECK-LABEL: @matmul_nn
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "cpu">, %[[ARG1:.*]]: memref<?x?xf32, "cpu">, %[[ARG2:.*]]: memref<?x?xf32, "cpu">)
  func.func @matmul_nn(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> memref<?x?xf32> attributes {test = true} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %0 = affine.apply #map()[%dim]
    %1 = affine.apply #map1()[%dim_0]
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    scf.foreach_thread (%arg3, %arg4) in (%0, %1) {
      %2 = affine.min #map2(%arg3)[%dim]
      %3 = affine.min #map3(%arg4)[%dim_0]
      %4 = affine.apply #map4(%arg3)
      %5 = affine.apply #map5(%arg4)
      %subview = memref.subview %arg0[%4, 0] [%2, %dim_1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[0, %5] [%dim_1, %3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%4, %5] [%2, %3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
      linalg.matmul ins(%subview, %subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
    } {thread_dim_mapping = []}
    return %arg2 : memref<?x?xf32>
  }
}