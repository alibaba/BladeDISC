// RUN: disc-opt -split-input-file -pass-pipeline='builtin.module(func.func(disc-fusion{gpu-enabled=false fusion-strategy=stitch}))' %s | FileCheck %s --check-prefix=SPARSE

// SPARSE-LABEL: @where_input_fusion
func.func @where_input_fusion(
        %arg0: memref<i64>,
        %arg1: memref<3xindex>,
        %arg2: memref<?x?x?xi64>,
        %arg3: memref<?x?x?xi64>,
        %arg4: memref<?x?x?xi1>,
        %arg5: memref<?x3xi64>,
        %arg6: memref<1xi64>
    ) -> (
    memref<?x3xi64>,
    memref<1xi64>
  ) {
  // SPARSE: "lmhlo.fusion"() ({
  // SPARSE-NEXT: lmhlo.dynamic_broadcast_in_dim
  // SPARSE-NEXT: lmhlo.compare
  // SPARSE-NEXT: lmhlo_disc.where
  // SPARSE-NEXT: lmhlo.terminator
  // SPARSE-NEXT: })
  // SPARSE-SAME: disc.fusion_type = "kWhere"
  // SPARSE-NEXT: return
  "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg2) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i64>, memref<3xindex>, memref<?x?x?xi64>) -> ()
  "lmhlo.compare"(%arg3, %arg2, %arg4) {comparison_direction = #mhlo<comparison_direction NE>, disc.device = "cpu"} : (memref<?x?x?xi64>, memref<?x?x?xi64>, memref<?x?x?xi1>) -> ()
  "lmhlo_disc.where"(%arg4, %arg5, %arg6) {disc.device = "cpu"} : (memref<?x?x?xi1>, memref<?x3xi64>, memref<1xi64>) -> ()
  return %arg5, %arg6 : memref<?x3xi64>, memref<1xi64>
}

// -----

// SPARSE-LABEL: @sparse_reduction_fusion
func.func @sparse_reduction_fusion(
        %arg0: memref<f32>,
        %arg1: memref<?x?xf32>,
        %arg2: memref<?xi64>,
        %arg3: memref<?x2xi64>,
        %arg4: memref<?xi64>,
        %arg5: memref<?x?xf32>,
        %arg6: memref<?xi1>,
        %arg7: memref<4xindex>,
        %arg8: memref<1x?x?x1xi1>,
        %arg9: memref<2xindex>,
        %arg10: memref<?x?xi1>,
        %arg11: memref<?x?xf32>,
        %arg12: memref<?x?xf32>
    ) -> (
        memref<?x?xf32>
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
  "lmhlo.constant"(%arg0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
  "lmhlo_disc.sparse_segment_reduction_with_empty_rows"(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {disc.device = "cpu", is_mean = false} : (memref<?x?xf32>, memref<?xi64>, memref<?x2xi64>, memref<?xi64>, memref<?x?xf32>, memref<?xi1>) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg6, %arg7, %arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "cpu"} : (memref<?xi1>, memref<4xindex>, memref<1x?x?x1xi1>) -> ()
  "lmhlo.dynamic_reshape"(%arg8, %arg9, %arg10) {disc.device = "cpu"} : (memref<1x?x?x1xi1>, memref<2xindex>, memref<?x?xi1>) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg9, %arg11) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<f32>, memref<2xindex>, memref<?x?xf32>) -> ()
  "lmhlo.select"(%arg10, %arg11, %arg5, %arg12) {disc.device = "cpu"} : (memref<?x?xi1>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg12 : memref<?x?xf32>
}
