// RUN: disc-opt --disc-legalize-lmhlo-fusion-to-linalg -split-input-file %s | FileCheck %s

// CHECK-LABEL: @matmul_nn
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @matmul_nn(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T1:.*]] = linalg.fill ins(%[[T0]] : f32) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T2:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[T1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[T2]]
  "lmhlo.fusion"() ({
    "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "matmul_nn_kTransform_dot_general__1_1_0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}