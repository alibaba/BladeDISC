// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-output-inline-fusion -split-input-file | FileCheck %s

// CHECK-LABEL: @where_fusion
func.func @where_fusion(
    %arg0: memref<?xi64, "cpu">,
    %arg1: memref<?xi1, "cpu">,
    %arg2: memref<?x1xi64, "cpu">,
    %arg3: memref<1xi64, "cpu">,
    %arg4: memref<?xi64, "cpu">,
    %arg5: memref<?xi64, "cpu">,
    %arg6: memref<?xi64, "cpu">,
    %arg7: memref<?x2xi64, "cpu">,
    %arg8: memref<?x2xi64, "cpu">
    ) -> (
    memref<?xi64, "cpu">,
    memref<?x2xi64, "cpu">,
    memref<1xi64, "cpu">
) {
  %false = arith.constant false
  %c2 = arith.constant 2 : index
  %c0_i64 = arith.constant 0 : i64
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
    %alloca = memref.alloca() : memref<i64, "cpu">
    memref.store %c0_i64, %alloca[] : memref<i64, "cpu">
    %dim = memref.dim %arg1, %c0 : memref<?xi1, "cpu">
    scf.parallel (%arg9) = (%c0) to (%dim) step (%c1) {
      %4 = memref.load %arg0[%arg9] : memref<?xi64, "cpu">
      %5 = arith.cmpi ne, %4, %c0_i64 : i64
      %6 = arith.cmpi ne, %5, %false : i1
      scf.if %6 {
        %1 = arith.index_cast %arg9 : index to i64
        %7 = memref.load %alloca[] : memref<i64, "cpu">
        %8 = arith.index_cast %7 : i64 to index
        %dim_2 = memref.dim %arg2, %c0 : memref<?x1xi64, "cpu">
        %dim_3 = memref.dim %arg4, %c0 : memref<?xi64, "cpu">
        %9 = "disc_shape.linearize"(%8, %c0, %dim_2, %c1) {operand_segment_sizes = array<i32: 2, 2>} : (index, index, index, index) -> index
        %10 = "disc_shape.delinearize"(%9, %dim_3) : (index, index) -> index
        memref.store %1, %arg4[%10] : memref<?xi64, "cpu">
        %11 = arith.addi %7, %c1_i64 : i64
        memref.store %11, %alloca[] : memref<i64, "cpu">
      }
      scf.yield
    }
    %0 = memref.load %alloca[] : memref<i64, "cpu">
    memref.store %0, %arg3[%c0] : memref<1xi64, "cpu">
    "lmhlo.dynamic_gather"(%arg5, %arg4, %alloca_6, %arg6) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, disc.device = "cpu", disc.shape_op = true, indices_are_sorted = false} : (memref<?xi64, "cpu">, memref<?xi64, "cpu">, memref<1xi64, "cpu">, memref<?xi64, "cpu">) -> ()
    "lmhlo.dynamic_gather"(%arg7, %arg4, %alloca_7, %arg8) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, disc.device = "cpu", disc.shape_op = true, indices_are_sorted = false} : (memref<?x2xi64, "cpu">, memref<?xi64, "cpu">, memref<2xi64, "cpu">, memref<?x2xi64, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "main_kWhere_where__4_2_0", disc.fusion_type = "kWhere"} : () -> ()
  return %arg6, %arg8, %arg3 : memref<?xi64, "cpu">, memref<?x2xi64, "cpu">, memref<1xi64, "cpu">
}
