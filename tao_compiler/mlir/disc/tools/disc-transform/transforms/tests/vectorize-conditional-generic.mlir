// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s --dump-input=always

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @vectorize_conditional_generic
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<12xf32>, %[[ARG3:.*]]: tensor<8x12xf32>)
func.func @vectorize_conditional_generic(
    %pred : i1, %arg0: tensor<f32>, %arg1 : tensor<12xf32>, %arg2 : tensor<8x12xf32>) -> tensor<8x12xf32> {
  // CHECK: %[[T0:.*]] = vector.transfer_read %[[ARG3]]
  // CHECK: %[[OUT:.*]] = scf.if %[[ARG0:.*]] -> (vector<8x12xf32>)
  // CHECK-NEXT:   %[[T1:.*]] = vector.transfer_read %[[ARG1]]
  // CHECK-NEXT:   %[[T2:.*]] = vector.transfer_read %[[ARG2]]
  // CHECK-NEXT:   %[[T3:.*]] = arith.addf %[[T0]], %[[T2]] : vector<8x12xf32>
  // CHECK-NEXT:   %[[T4:.*]] = arith.maxf %[[T1]], %[[T3]] : vector<8x12xf32>
  // CHECK-NEXT:   scf.yield %[[T4]] : vector<8x12xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %0 : vector<8x12xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[UPDATE_OUT:.*]] = vector.transfer_write %[[OUT]], %[[ARG3]]
  // CHECK-NEXT: return %[[UPDATE_OUT]]
  %out = disc_linalg_ext.conditional_generic {indexing_maps = [#map0, #map0, #map1, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%pred, %arg0, %arg1 : i1, tensor<f32>, tensor<12xf32>) outs(%arg2 : tensor<8x12xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "maximum"} {
  ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
    %t0 = arith.addf %out, %in_2 : f32
    %t1 = arith.maxf %in_1, %t0 : f32
    disc_linalg_ext.yield %t1 : f32
  } -> tensor<8x12xf32>
  return %out : tensor<8x12xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["disc_linalg_ext.conditional_generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.disc.vectorize_conditional_generic %0
}