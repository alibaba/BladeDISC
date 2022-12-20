// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-output-inline-fusion -split-input-file | FileCheck %s

// CHECK-LABEL: @where_fusion
func.func @where_fusion(
    %arg0: memref<?x?x?xi64, "cpu">,
    %arg1: memref<?x?x?xi1, "cpu">,
    %arg2: memref<?x3xi64, "cpu">,
    %arg3: memref<1xi64, "cpu">,
    %arg4: memref<?xi64, "cpu">,
    %arg5: memref<1xindex, "cpu">
    ) -> (
    memref<?xi64, "cpu">,
    memref<1xi64, "cpu">
) {
  %false = arith.constant false
  %c2 = arith.constant 2 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT lmhlo.dynamic_reshape
  "lmhlo.fusion"() ({
    %alloca = memref.alloca() : memref<i64, "cpu">
    memref.store %c0_i64, %alloca[] : memref<i64, "cpu">
    %dim = memref.dim %arg1, %c0 : memref<?x?x?xi1, "cpu">
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?x?xi1, "cpu">
    %dim_1 = memref.dim %arg1, %c2 : memref<?x?x?xi1, "cpu">
    scf.parallel (%arg6, %arg7, %arg8) = (%c0, %c0, %c0) to (%dim, %dim_0, %dim_1) step (%c1, %c1, %c1) {
      %4 = memref.load %arg0[%arg6, %arg7, %arg8] : memref<?x?x?xi64, "cpu">
      %5 = arith.cmpi ne, %4, %c0_i64 : i64
      %6 = arith.cmpi ne, %5, %false : i1
      scf.if %6 {
        %1 = arith.index_cast %arg6 : index to i64
        %2 = arith.index_cast %arg7 : index to i64
        %3 = arith.index_cast %arg8 : index to i64
        %7 = memref.load %alloca[] : memref<i64, "cpu">
        %8 = arith.index_cast %7 : i64 to index
        memref.store %1, %arg2[%8, %c0] : memref<?x3xi64, "cpu">
        memref.store %2, %arg2[%8, %c1] : memref<?x3xi64, "cpu">
        memref.store %3, %arg2[%8, %c2] : memref<?x3xi64, "cpu">
        %9 = arith.addi %7, %c1_i64 : i64
        memref.store %9, %alloca[] : memref<i64, "cpu">
      }
      scf.yield
    }
    %0 = memref.load %alloca[] : memref<i64, "cpu">
    memref.store %0, %arg3[%c0] : memref<1xi64, "cpu">
    "lmhlo.dynamic_reshape"(%arg2, %arg5, %arg4) : (memref<?x3xi64, "cpu">, memref<1xindex, "cpu">, memref<?xi64, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "main_kWhere_where__4_2_0", disc.fusion_type = "kWhere"} : () -> ()
  return %arg4, %arg3 : memref<?xi64, "cpu">, memref<1xi64, "cpu">
}
