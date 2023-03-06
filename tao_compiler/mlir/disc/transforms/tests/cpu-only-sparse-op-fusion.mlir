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
