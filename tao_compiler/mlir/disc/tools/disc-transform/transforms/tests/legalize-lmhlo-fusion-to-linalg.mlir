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