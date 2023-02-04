// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @elemwise_fuse
func.func @elemwise_fuse(%arg0: tensor<?x3072xf32>, %arg1: tensor<2xindex>, %arg2: tensor<3072x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<?x768xf32>, %arg5: tensor<?x768xf32>, %arg6: tensor<?x768xf32>, %arg7: tensor<?x768xf32>) -> tensor<?x768xf32> {
  // CHECK: disc_linalg_ext.constant_wrapper
  // CHECK-NEXT: disc_linalg_ext.constant_wrapper
  // CHECK-NEXT: linalg.fill
  // CHECK-NEXT: linalg.matmul
  // CHECK-NEXT: %[[RES:.*]] = linalg.generic
  // CHECK-NEXT: (%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32)
  // CHECK-NEXT: %[[T0:.*]] = arith.subf %[[IN0]], %[[IN1]] : f32
  // CHECK-NEXT: %[[T1:.*]] = arith.addf %[[T0]], %[[T0]] : f32
  // CHECK-NEXT: linalg.yield %[[T1]] : f32
  // CHECK: return %[[RES]]
  %0 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant"} dense<-8.000000e-01> : tensor<3072x768xf32>
  %1 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant_1"} dense<-1.000000e-01> : tensor<768xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.fill {disc.transform.name = "dot_general"} ins(%cst : f32) outs(%arg4 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %3 = linalg.matmul {disc.transform.name = "dot_general"} ins(%arg0, %0 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%2 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<768xf32>) outs(%arg5 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "dynamic_broadcast_in_dim"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x768xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%arg6 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "subtract"} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<?x768xf32>
  %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5, %5 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%arg7 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "add"} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<?x768xf32>
  return %6 : tensor<?x768xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {disc.transform.name = "dynamic_broadcast_in_dim"} in %arg0
  %1 = transform.structured.match attributes {disc.transform.name = "subtract"} in %arg0
  %2 = transform.structured.match attributes {disc.transform.name = "add"} in %arg0
  %3 = transform.disc.linalg.fuse_producers %0, %1 into %2
}
