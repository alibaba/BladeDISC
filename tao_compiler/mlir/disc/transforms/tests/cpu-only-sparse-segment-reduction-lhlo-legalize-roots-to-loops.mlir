// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @sparse_segment_mean
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xf32, "cpu">, %[[INPUT2:.*]]: memref<?xi32, "cpu">, %[[INPUT3:.*]]: memref<?xi32, "cpu">, %[[OUT1:.*]]: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu">
func.func @sparse_segment_mean(
    %input1: memref<?x?xf32, "cpu">,
    %input2: memref<?xi32, "cpu">,
    %input3: memref<?xi32, "cpu">,
    %out1: memref<?x?xf32, "cpu">
    ) -> (
    memref<?x?xf32, "cpu">
  ) {
  // CHECK-NOT lmhlo
  // CHECK: scf.parallel (%[[ARG4:.*]], %[[ARG5:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK:   %[[V1:.*]] = memref.load %{{.*}}[%[[ARG4]]] : memref<?xi32, "cpu">
  // CHECK:   %[[V2:.*]] = arith.index_cast %[[V1]] : i32 to index
  // CHECK:   %[[V3:.*]] = memref.load %{{.*}}[%[[V2]]] : memref<?xf32, "cpu">
  // CHECK:   %[[V4:.*]] = arith.cmpf one, %[[V3]], %{{.*}} : f32
  // CHECK:   scf.if %[[V4]] {
  // CHECK:     %[[V5:.*]] = memref.load %{{.*}}[%[[ARG4]]] : memref<?xi32, "cpu">
  // CHECK:     %[[V6:.*]] = arith.index_cast %[[V5]] : i32 to index
  // CHECK:     %[[V7:.*]] = arith.index_cast %[[V1]] : i32 to index
  // CHECK:     %[[V8:.*]] = memref.load %{{.*}}[%[[V6]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:     %[[V9:.*]] = arith.divf %[[V8]], %[[V3]] : f32
  // CHECK:     %[[V10:.*]] = memref.load %{{.*}}[%[[V7]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:     %[[V11:.*]] = arith.addf %[[V10]], %[[V9]] : f32
  // CHECK:     memref.store %[[V11]], %{{.*}}[%[[V7]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:   }
  // CHECK: }
  "lmhlo_disc.sparse_segment_reduction"(%input1, %input2, %input3, %out1) {disc.device = "cpu", reduction_mode = 0} : (memref<?x?xf32, "cpu">, memref<?xi32, "cpu">, memref<?xi32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  // CHECK: return %[[OUT1]] : memref<?x?xf32, "cpu">
  return %out1 : memref<?x?xf32, "cpu">
}

// -----

// CHECK-LABEL: @sparse_segment_sum
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xf32, "cpu">, %[[INPUT2:.*]]: memref<?xi32, "cpu">, %[[INPUT3:.*]]: memref<?xi32, "cpu">, %[[OUT1:.*]]: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu">
func.func @sparse_segment_sum(
    %input1: memref<?x?xf32, "cpu">,
    %input2: memref<?xi32, "cpu">,
    %input3: memref<?xi32, "cpu">,
    %out1: memref<?x?xf32, "cpu">
    ) -> (
    memref<?x?xf32, "cpu">
  ) {
  // CHECK-NOT lmhlo
  // CHECK: scf.parallel (%[[ARG4:.*]], %[[ARG5:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK:   %[[V1:.*]] = memref.load %{{.*}}[%[[ARG4]]] : memref<?xi32, "cpu">
  // CHECK:   %[[V2:.*]] = arith.index_cast %[[V1]] : i32 to index
  // CHECK:   %[[V3:.*]] = memref.load %{{.*}}[%[[ARG4]]] : memref<?xi32, "cpu">
  // CHECK:   %[[V4:.*]] = arith.index_cast %[[V3]] : i32 to index
  // CHECK-DAG:   %[[V5:.*]] = memref.load %{{.*}}[%[[V2]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK-DAG:   %[[V6:.*]] = memref.load %{{.*}}[%[[V4]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:   %[[V7:.*]] = arith.addf %[[V6]], %[[V5]] : f32
  // CHECK:   memref.store %[[V7]], %{{.*}}[%[[V4]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:   scf.yield
  // CHECK: }
  "lmhlo_disc.sparse_segment_reduction"(%input1, %input2, %input3, %out1) {disc.device = "cpu", reduction_mode = 1} : (memref<?x?xf32, "cpu">, memref<?xi32, "cpu">, memref<?xi32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  // CHECK: return %[[OUT1]] : memref<?x?xf32, "cpu">
  return %out1 : memref<?x?xf32, "cpu">
}

// -----

// CHECK-LABEL: @sparse_segment_mean_with_empty_rows
func.func @sparse_segment_mean_with_empty_rows(
    %arg0: memref<?x?xf32, "cpu">,
    %arg1: memref<?xi64, "cpu">,
    %arg2: memref<?x2xi64, "cpu">,
    %arg3: memref<?xi64, "cpu">,
    %output: memref<?x?xf32, "cpu">,
    %empty_row_indicator: memref<?xi1, "cpu">)
-> (memref<?x?xf32, "cpu">,
    memref<?xi1, "cpu">) {
  // CHECK: scf.parallel
  // CHECK: scf.if %{{.*}} {
  // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xi1, "cpu">
  // CHECK-DAG:   %[[V7:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK-DAG:   %[[V8:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf32, "cpu">
  // CHECK-DAG:   %[[V9:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK-DAG:   %[[V10:.*]] = arith.divf %[[V9]], %[[V8]] : f32
  // CHECK-DAG:   %[[V11:.*]] = arith.addf %[[V7]], %[[V10]] : f32
  // CHECK:   memref.store %[[V11]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK: } else {
  // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xi1, "cpu">
  // CHECK:   %[[V7:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK:   memref.store %[[V7]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK: }
  "lmhlo_disc.sparse_segment_reduction_with_empty_rows"(%arg0, %arg1, %arg2, %arg3, %output, %empty_row_indicator) {reduction_mode = 0 : i64} : (memref<?x?xf32, "cpu">, memref<?xi64, "cpu">, memref<?x2xi64, "cpu">, memref<?xi64, "cpu">, memref<?x?xf32, "cpu">, memref<?xi1, "cpu">) -> ()
  return %output, %empty_row_indicator : memref<?x?xf32, "cpu">, memref<?xi1, "cpu">
}

// -----

// CHECK-LABEL: @sparse_segment_sum_with_empty_rows
func.func @sparse_segment_sum_with_empty_rows(
    %arg0: memref<?x?xf32, "cpu">,
    %arg1: memref<?xi64, "cpu">,
    %arg2: memref<?x2xi64, "cpu">,
    %arg3: memref<?xi64, "cpu">,
    %output: memref<?x?xf32, "cpu">,
    %empty_row_indicator: memref<?xi1, "cpu">)
-> (memref<?x?xf32, "cpu">,
    memref<?xi1, "cpu">) {
  // CHECK: scf.parallel
  // CHECK: scf.if %{{.*}} {
  // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xi1, "cpu">
  // CHECK-DAG:   %[[V7:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK-DAG:   %[[V8:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK-DAG:   %[[V9:.*]] = arith.addf %[[V7]], %[[V8]] : f32
  // CHECK:   memref.store %[[V9]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK: } else {
  // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xi1, "cpu">
  // CHECK:   %[[V7:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK:   memref.store %[[V7]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, "cpu">
  // CHECK: }
  "lmhlo_disc.sparse_segment_reduction_with_empty_rows"(%arg0, %arg1, %arg2, %arg3, %output, %empty_row_indicator) {reduction_mode = 1 : i64} : (memref<?x?xf32, "cpu">, memref<?xi64, "cpu">, memref<?x2xi64, "cpu">, memref<?xi64, "cpu">, memref<?x?xf32, "cpu">, memref<?xi1, "cpu">) -> ()
  return %output, %empty_row_indicator : memref<?x?xf32, "cpu">, memref<?xi1, "cpu">
}
