// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @sparse_fill_empty_rows
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xi64, "cpu">, %[[INPUT2:.*]]: memref<?xi64, "cpu">, %[[INPUT3:.*]]: memref<?xi64, "cpu">, %[[INPUT4:.*]]: memref<i64, "cpu">, %[[OUT1:.*]]: memref<?x?xi64, "cpu">, %[[OUT2:.*]]: memref<?xi64, "cpu">, %[[OUT3:.*]]: memref<?xi1, "cpu">, %[[OUT4:.*]]: memref<?xi64, "cpu">, %[[OUT5:.*]]: memref<?xi64, "cpu">) -> (memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi1, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">)
func.func @sparse_fill_empty_rows(
    %input1: memref<?x?xi64, "cpu">,
    %input2: memref<?xi64, "cpu">,
    %input3: memref<?xi64, "cpu">,
    %input4: memref<i64, "cpu">,
    %out1: memref<?x?xi64, "cpu">,
    %out2: memref<?xi64, "cpu">,
    %out3: memref<?xi1, "cpu">,
    %out4: memref<?xi64, "cpu">,
    %out5: memref<?xi64, "cpu">
    ) -> (
    memref<?x?xi64, "cpu">,
    memref<?xi64, "cpu">,
    memref<?xi1, "cpu">,
    memref<?xi64, "cpu">,
    memref<?xi64, "cpu">
  ) {
  "lmhlo_disc.sparse_fill_empty_rows"(%input1, %input2, %input3, %input4, %out1, %out2, %out3, %out4, %out5) : (memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">, memref<i64, "cpu">, memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi1, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]], %[[OUT3]], %[[OUT4]], %[[OUT5]] : memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi1, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">
  return %out1, %out2, %out3, %out4, %out5 : memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi1, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">
}
