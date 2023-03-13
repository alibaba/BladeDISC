// RUN: disc-opt -split-input-file -pass-pipeline='builtin.module(func.func(disc-fusion{gpu-enabled=false fusion-strategy=stitch}))' %s | FileCheck %s --check-prefix=SPARSE

// SPARSE-LABEL: @where_input_fusion
func.func @where_input_fusion(
        %arg0: memref<i64, "cpu">,
        %arg1: memref<3xindex, "cpu">,
        %arg2: memref<?x?x?xi64, "cpu">,
        %arg3: memref<?x?x?xi64, "cpu">,
        %arg4: memref<?x?x?xi1, "cpu">,
        %arg5: memref<?x3xi64, "cpu">,
        %arg6: memref<1xi64, "cpu">
    ) -> (
    memref<?x3xi64, "cpu">,
    memref<1xi64, "cpu">
  ) {
  // SPARSE: "lmhlo.fusion"() ({
  // SPARSE-NEXT: lmhlo.dynamic_broadcast_in_dim
  // SPARSE-NEXT: lmhlo.compare
  // SPARSE-NEXT: lmhlo_disc.where
  // SPARSE-NEXT: lmhlo.terminator
  // SPARSE-NEXT: })
  // SPARSE-SAME: disc.fusion_type = "kWhere"
  // SPARSE-NEXT: return
  "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg2) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i64, "cpu">, memref<3xindex, "cpu">, memref<?x?x?xi64, "cpu">) -> ()
  "lmhlo.compare"(%arg3, %arg2, %arg4) {comparison_direction = #mhlo<comparison_direction NE>, disc.device = "cpu"} : (memref<?x?x?xi64, "cpu">, memref<?x?x?xi64, "cpu">, memref<?x?x?xi1, "cpu">) -> ()
  "lmhlo_disc.where"(%arg4, %arg5, %arg6) {disc.device = "cpu"} : (memref<?x?x?xi1, "cpu">, memref<?x3xi64, "cpu">, memref<1xi64, "cpu">) -> ()
  return %arg5, %arg6 : memref<?x3xi64, "cpu">, memref<1xi64, "cpu">
}

// -----

// SPARSE-LABEL: @sparse_reduction_fusion
func.func @sparse_reduction_fusion(
        %arg0: memref<f32, "cpu">,
        %arg1: memref<?x?xf32, "cpu">,
        %arg2: memref<?xi64, "cpu">,
        %arg3: memref<?x2xi64, "cpu">,
        %arg4: memref<?xi64, "cpu">,
        %arg5: memref<?x?xf32, "cpu">,
        %arg6: memref<?xi1, "cpu">,
        %arg7: memref<4xindex, "cpu">,
        %arg8: memref<1x?x?x1xi1, "cpu">,
        %arg9: memref<2xindex, "cpu">,
        %arg10: memref<?x?xi1, "cpu">,
        %arg11: memref<?x?xf32, "cpu">,
        %arg12: memref<?x?xf32, "cpu">
    ) -> (
        memref<?x?xf32, "cpu">
  ) {
  // SPARSE: "lmhlo.fusion"() ({
  // SPARSE-NEXT: lmhlo.constant
  // SPARSE-NEXT: lmhlo_disc.sparse_segment_reduction_with_empty_rows
  // SPARSE-NEXT: lmhlo.dynamic_broadcast_in_dim
  // SPARSE-NEXT: lmhlo.dynamic_reshape
  // SPARSE-NEXT: lmhlo.dynamic_broadcast_in_dim
  // SPARSE-NEXT: lmhlo.select
  // SPARSE-NEXT: lmhlo.terminator
  // SPARSE-NEXT: })
  // SPARSE-SAME: disc.fusion_type = "kSparseReduction"
  // SPARSE-NEXT: return
  "lmhlo.constant"(%arg0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  "lmhlo_disc.sparse_segment_reduction_with_empty_rows"(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {disc.device = "cpu", is_mean = false} : (memref<?x?xf32, "cpu">, memref<?xi64, "cpu">, memref<?x2xi64, "cpu">, memref<?xi64, "cpu">, memref<?x?xf32, "cpu">, memref<?xi1, "cpu">) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg6, %arg7, %arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "cpu"} : (memref<?xi1, "cpu">, memref<4xindex, "cpu">, memref<1x?x?x1xi1, "cpu">) -> ()
  "lmhlo.dynamic_reshape"(%arg8, %arg9, %arg10) {disc.device = "cpu"} : (memref<1x?x?x1xi1, "cpu">, memref<2xindex, "cpu">, memref<?x?xi1, "cpu">) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg9, %arg11) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<f32, "cpu">, memref<2xindex, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo.select"(%arg10, %arg11, %arg5, %arg12) {disc.device = "cpu"} : (memref<?x?xi1, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  return %arg12 : memref<?x?xf32, "cpu">
}
