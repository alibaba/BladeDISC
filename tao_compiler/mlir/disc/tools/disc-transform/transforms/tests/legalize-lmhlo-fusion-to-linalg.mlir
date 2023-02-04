// RUN: disc-opt --disc-legalize-lmhlo-fusion-to-linalg -split-input-file %s | FileCheck %s

// CHECK-LABEL: @matmul_nn
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @matmul_nn(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T1:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T0]] : f32) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T2:.*]] = linalg.matmul {disc.transform.name = "dot_general"} ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[T1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[T2]]
  "lmhlo.fusion"() ({
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// CHECK-LABEL: @packed_matmul_nn
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x512xf32>, %[[ARG1:.*]]: tensor<512x1024xf32>, %[[ARG2:.*]]: tensor<?x1024xf32>)
func.func @packed_matmul_nn(%arg1: memref<?x512xf32, "cpu">, %arg2: memref<512x1024xf32, "cpu">, %arg3: memref<?x1024xf32, "cpu">) -> memref<?x1024xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant"} dense<-8.000000e-01> : tensor<512x1024xf32>
  // CHECK: %[[T1:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T2:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T1]] : f32) outs(%[[ARG2]] : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  // CHECK: %[[T3:.*]] = linalg.matmul {disc.transform.name = "dot_general"} ins(%[[ARG0]], %[[T0]] : tensor<?x512xf32>, tensor<512x1024xf32>) outs(%[[T2]] : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  // CHECK: return %[[T3]]
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%arg2) {disc.device = "cpu", value = dense<-8.000000e-01> : tensor<512x1024xf32>} : (memref<512x1024xf32, "cpu">) -> ()
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {disc.device = "cpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x512xf32, "cpu">, memref<512x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x1024xf32, "cpu">
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @matmul_tn
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @matmul_tn(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T1:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T0]] : f32) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T2:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[T1]]
  // CHECK-SAME: {disc.transform.name = "dot_general"}
  // CHECK: return %[[T2]]
  "lmhlo.fusion"() ({
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @matmul_nt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @matmul_nt(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T1:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T0]] : f32) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T2:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[T1]]
  // CHECK-SAME: {disc.transform.name = "dot_general"}
  // CHECK: return %[[T2]]
  "lmhlo.fusion"() ({
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @matmul_tt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @matmul_tt(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T1:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T0]] : f32) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T2:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[T1]]
  // CHECK-SAME: {disc.transform.name = "dot_general"}
  // CHECK: return %[[T2]]
  "lmhlo.fusion"() ({
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @matmul_epilogue
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x3072xf32>, %[[ARG1:.*]]: tensor<2xindex>, %[[ARG2:.*]]: tensor<3072x768xf32>,
// CHECK-SAME: %[[ARG3:.*]]: tensor<768xf32>, %[[ARG4:.*]]: tensor<?x768xf32>, %[[ARG5:.*]]: tensor<?x768xf32>,
// CHECK-SAME: %[[ARG6:.*]]: tensor<?x768xf32>)
func.func @matmul_epilogue(%arg0: memref<?x3072xf32, "cpu">, %arg1: memref<2xindex, "cpu">, %arg2: memref<3072x768xf32, "cpu">, %arg3: memref<768xf32, "cpu">, %arg4: memref<?x768xf32, "cpu">, %arg5: memref<?x768xf32, "cpu">, %arg6: memref<?x768xf32, "cpu">) -> memref<?x768xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant"} dense<-8.000000e-01> : tensor<3072x768xf32>
  // CHECK: %[[T1:.*]] = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant_1"} dense<-1.000000e-01> : tensor<768xf32>
  // CHECK: %[[T2:.*]] = arith.constant 0.000000e+00 : f32

  // CHECK: %[[T3:.*]] = linalg.fill {disc.transform.name = "dot_general"} ins(%[[T2]] : f32) outs(%[[ARG4]] : tensor<?x768xf32>) -> tensor<?x768xf32>
  // CHECK: %[[T4:.*]] = linalg.matmul {disc.transform.name = "dot_general"} ins(%[[ARG0]], %[[T0]] : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%[[T3]] : tensor<?x768xf32>) -> tensor<?x768xf32>

  // CHECK: %[[T5:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[T1]] : tensor<768xf32>) outs(%[[ARG5]] : tensor<?x768xf32>)
  // CHECK-SAME: disc.transform.name = "dynamic_broadcast_in_dim"

  // CHECK: %[[T6:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map1, #map1, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[T4]], %[[T5]] : tensor<?x768xf32>, tensor<?x768xf32>) outs(%[[ARG6]] : tensor<?x768xf32>)
  // CHECK-SAME: disc.transform.name = "subtract"

  // CHECK: return %[[T6]]

  "lmhlo.fusion"() ({
    "lmhlo.constant"(%arg2) {disc.device = "cpu", value = dense<-8.000000e-01> : tensor<3072x768xf32>} : (memref<3072x768xf32, "cpu">) -> ()
    "lmhlo.constant"(%arg3) {disc.device = "cpu", value = dense<-1.000000e-01> : tensor<768xf32>} : (memref<768xf32, "cpu">) -> ()
    "lmhlo.dot_general"(%arg0, %arg2, %arg4) {disc.device = "cpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x3072xf32, "cpu">, memref<3072x768xf32, "cpu">, memref<?x768xf32, "cpu">) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%arg3, %arg1, %arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "cpu"} : (memref<768xf32, "cpu">, memref<2xindex, "cpu">, memref<?x768xf32, "cpu">) -> ()
    "lmhlo.subtract"(%arg4, %arg5, %arg6) {disc.device = "cpu"} : (memref<?x768xf32, "cpu">, memref<?x768xf32, "cpu">, memref<?x768xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "test", disc.fusion_type = "kTransform"} : () -> ()
  return %arg6 : memref<?x768xf32, "cpu">
}
