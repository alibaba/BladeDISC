// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @lower_conditional_generic
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<12xf32>, %[[ARG3:.*]]: memref<8x12xf32>)
func.func @lower_conditional_generic(
    %pred : i1, %arg0: memref<f32>, %arg1 : memref<12xf32>, %arg2 : memref<8x12xf32>) -> memref<8x12xf32> {
  // CHECK: scf.if %[[ARG0]] {
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME: ins(%[[ARG1]], %[[ARG2]] : memref<f32>, memref<12xf32>)
  // CHECK-SAME: outs(%[[ARG3]] : memref<8x12xf32>)
  disc_linalg_ext.conditional_generic {indexing_maps = [#map0, #map0, #map1, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%pred, %arg0, %arg1 : i1, memref<f32>, memref<12xf32>) outs(%arg2 : memref<8x12xf32>) {
  ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
    %t0 = arith.addf %out, %in_2 : f32
    %t1 = arith.maxf %in_1, %t0 : f32
    disc_linalg_ext.yield %t1 : f32
  }
  return %arg2 : memref<8x12xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["disc_linalg_ext.conditional_generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.disc.lower_conditional_generic %0
}