// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @sparse_reshape_basic
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xi64, "cpu">, %[[INPUT2:.*]]: memref<?xi64, "cpu">, %[[INPUT3:.*]]: memref<?xi64, "cpu">, %[[OUT1:.*]]: memref<?x?xi64, "cpu">, %[[OUT2:.*]]: memref<?xi64, "cpu">) -> (memref<?x?xi64, "cpu">, memref<?xi64, "cpu">)
func.func @sparse_reshape_basic(
    %input1: memref<?x?xi64, "cpu">,
    %input2: memref<?xi64, "cpu">,
    %input3: memref<?xi64, "cpu">,
    %out1: memref<?x?xi64, "cpu">,
    %out2: memref<?xi64, "cpu">
    ) -> (
    memref<?x?xi64, "cpu">,
    memref<?xi64, "cpu">
  ) {
  // CHECK-NOT lmhlo
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[V0:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xi64, "cpu">
  // CHECK: %[[V1:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xi64, "cpu">
  // CHECK: %[[V2:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xi64, "cpu">
  // CHECK: %[[V3:.*]] = memref.alloc(%[[V2]]) : memref<?xi64>
  // CHECK: %[[V4:.*]] = arith.subi %[[V2]], %[[C1]] : index
  // CHECK: %[[V5:.*]] = memref.load %{{.*}}[%[[V4]]] : memref<?xi64, "cpu">
  // CHECK: scf.for %[[ARG5:.*]] = %[[C0]] to %[[V0]] step %[[C1]] {
  // CHECK:   %[[V7:.*]] = memref.load %{{.*}}[%[[ARG5]], %[[C0]]]  : memref<?x?xi64, "cpu">
  // CHECK:   %[[V8:.*]] = scf.for %[[ARG6:.*]] = %[[C1]] to %[[V1]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[V7]]) -> (i64) {
  // CHECK:     %[[V11:.*]] = memref.load %{{.*}}[%[[ARG5]], %[[ARG6]]] : memref<?x?xi64, "cpu">
  // CHECK:     %[[V12:.*]] = memref.load %{{.*}}[%[[ARG6]]] : memref<?xi64, "cpu">
  // CHECK:     %[[V13:.*]] = arith.muli %[[ARG7]], %[[V12]] : i64
  // CHECK:     %[[V14:.*]] = arith.addi %[[V13]], %[[V11]] : i64
  // CHECK:     scf.yield %[[V14]] : i64
  // CHECK:   }
  // CHECK:   %[[V9:.*]] = arith.subi %[[V2]], %[[C1]] : index
  // CHECK:   %[[V10:.*]] = scf.for %[[ARG6:.*]] = %[[C0]] to %[[V9]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[V8]]) -> (i64) {
  // CHECK:     %[[V11:.*]] = memref.load %[[V3]][%[[ARG6]]] : memref<?xi64>
  // CHECK:     %[[V12:.*]] = arith.divui %[[ARG7]], %[[V11]] : i64
  // CHECK:     %[[V13:.*]] = arith.remui %[[ARG7]], %[[V11]] : i64
  // CHECK:     memref.store %[[V12]], %{{.*}}[%[[ARG5]], %[[ARG6]]] : memref<?x?xi64, "cpu">
  // CHECK:     scf.yield %[[V13]] : i64
  // CHECK:   }
  "lmhlo_disc.sparse_reshape"(%input1, %input2, %input3, %out1, %out2) : (memref<?x?xi64, "cpu">, memref<?xi64, "cpu">, memref<?xi64, "cpu">, memref<?x?xi64, "cpu">, memref<?xi64, "cpu">) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?xi64, "cpu">, memref<?xi64, "cpu">
  return %out1, %out2 : memref<?x?xi64, "cpu">, memref<?xi64, "cpu">
}
