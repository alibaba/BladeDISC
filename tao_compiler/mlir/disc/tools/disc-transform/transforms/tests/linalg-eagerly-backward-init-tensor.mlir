// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x3072xf32>, %[[ARG4:.*]]: tensor<?x768xf32>, %[[ARG6:.*]]: tensor<?x768xf32>)
func.func @main(%arg0: tensor<?x3072xf32>, %arg4: tensor<?x768xf32>, %arg6: tensor<?x768xf32>) -> tensor<?x768xf32> {
  %0 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant"} dense<-8.000000e-01> : tensor<3072x768xf32>
  %1 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant_1"} dense<-1.000000e-01> : tensor<768xf32>
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T0:.*]] = linalg.fill
  // CHECK-SAME: outs(%[[ARG6]] : tensor<?x768xf32>)
  // CHECK-NEXT: %[[T1:.*]] = linalg.matmul
  // CHECK-SAME: outs(%[[T0]] : tensor<?x768xf32>)
  // CHECK-NEXT: %[[T2:.*]] = linalg.generic
  // CHECK-SAME: outs(%[[T1]] : tensor<?x768xf32>)
  // CHECK: return %[[T2]]
  %2 = linalg.fill {disc.transform.name = "dot_general"} ins(%cst : f32) outs(%arg4 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %3 = linalg.matmul {disc.transform.name = "dot_general"} ins(%arg0, %0 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%2 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %1 : tensor<?x768xf32>, tensor<768xf32>) outs(%arg6 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "subtract"} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<?x768xf32>
  return %4 : tensor<?x768xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {disc.transform.name = "subtract"} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.disc.linalg.eagerly_backward_init_tensor %0
}
