// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @where_fusion
func.func @where_fusion(
    %input1: memref<i64, "cpu">,
    %input2: memref<1xindex, "cpu">,
    %input3: memref<?xi64, "cpu">,
    %input4: memref<?xi64, "cpu">,
    %input5: memref<?xi1, "cpu">,
    %input6: memref<?x1xi64, "cpu">,
    %input7: memref<1xindex, "cpu">,
    %input8: memref<?xi64, "cpu">,
    %input9: memref<?x2xi64, "cpu">,
    %input10: memref<?xi64, "cpu">,
    %out1: memref<1xi64, "cpu">,
    %out2: memref<?x2xi64, "cpu">,
    %out3: memref<?xi64, "cpu">
    ) -> (
    memref<1xi64, "cpu">,
    memref<?x2xi64, "cpu">,
    memref<?xi64, "cpu">
  ) {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca_6 = memref.alloca() : memref<1xi64, "cpu">
    memref.store %c1_i64, %alloca_6[%c0] : memref<1xi64, "cpu">
    %alloca_7 = memref.alloca() : memref<2xi64, "cpu">
    memref.store %c1_i64, %alloca_7[%c0] : memref<2xi64, "cpu">
    memref.store %c2_i64, %alloca_7[%c1] : memref<2xi64, "cpu">
    "lmhlo.fusion"() ({
      "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %input3) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i64, "cpu">, memref<1xindex, "cpu">, memref<?xi64, "cpu">) -> ()
      "lmhlo.compare"(%input4, %input3, %input5) {comparison_direction = #mhlo<comparison_direction GE>, disc.device = "cpu"} : (memref<?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi1, "cpu">) -> ()
      "lmhlo_disc.where"(%input5, %input6, %out1) {disc.device = "cpu"} : (memref<?xi1, "cpu">, memref<?x1xi64, "cpu">, memref<1xi64, "cpu">) -> ()
      "lmhlo.dynamic_reshape"(%input6, %input7, %input8) {disc.device = "cpu", disc.shape_op = true} : (memref<?x1xi64, "cpu">, memref<1xindex, "cpu">, memref<?xi64, "cpu">) -> ()
      "lmhlo.dynamic_gather"(%input9, %input8, %alloca_7, %out2) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, disc.device = "cpu", disc.shape_op = true, indices_are_sorted = false} : (memref<?x2xi64, "cpu">, memref<?xi64, "cpu">, memref<2xi64, "cpu">, memref<?x2xi64, "cpu">) -> ()
      "lmhlo.dynamic_gather"(%input10, %input8, %alloca_6, %out3) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, disc.device = "cpu", disc.shape_op = true, indices_are_sorted = false} : (memref<?xi64, "cpu">, memref<?xi64, "cpu">, memref<1xi64, "cpu">, memref<?xi64, "cpu">) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {disc.device = "cpu", disc.fusion.name = "main_kWhere_where_dynamic_gather_dynamic_gather__6_3_0", disc.fusion_type = "kWhere"} : () -> ()
  return %out1, %out2, %out3 : memref<1xi64, "cpu">, memref<?x2xi64, "cpu">, memref<?xi64, "cpu">
}
