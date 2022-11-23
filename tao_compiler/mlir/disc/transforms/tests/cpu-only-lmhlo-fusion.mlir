// RUN: disc-opt -pass-pipeline='func.func(disc-fusion{gpu-enabled=false fusion-strategy=base})' -split-input-file %s -o - | FileCheck %s --check-prefix=BASE
// RUN: DISC_ENABLE_TRANSFORM_SCHEDULE=1 disc-opt -pass-pipeline='func.func(disc-fusion{gpu-enabled=false fusion-strategy=stitch})' -split-input-file %s -o - | FileCheck %s --check-prefix=TRANSFORM

// BASE-LABEL: @custom_call_op
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "cpu">, %[[ARG1:.*]]: memref<?x?xf32, "cpu">, %[[ARG2:.*]]: memref<?x?xf32, "cpu">, %[[ARG3:.*]]: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu">
func.func @custom_call_op(%arg0: memref<?x?xf32, "cpu">, %arg1: memref<?x?xf32, "cpu">,
                     %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // BASE-NOT: "lmhlo.fusion"
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo_disc.custom_call"(%arg1, %arg2) {backend_config = "{}", call_target_name = "test", disc.device = "cpu", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// TRANSFORM-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: memref<?x?xf32, "cpu">, %arg1: memref<?x?xf32, "cpu">,
                          %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // TRANSFORM: "lmhlo.fusion"() ({
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.terminator
  // TRANSFORM-NEXT: })
  // TRANSFORM-SAME: disc.fusion_type = "kTransform"
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}