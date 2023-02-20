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
  "lmhlo_disc.sparse_segment_reduction"(%input1, %input2, %input3, %out1) {disc.device = "cpu", is_mean = true} : (memref<?x?xf32, "cpu">, memref<?xi32, "cpu">, memref<?xi32, "cpu">, memref<?x?xf32, "cpu">) -> ()
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
  "lmhlo_disc.sparse_segment_reduction"(%input1, %input2, %input3, %out1) {disc.device = "cpu", is_mean = false} : (memref<?x?xf32, "cpu">, memref<?xi32, "cpu">, memref<?xi32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  // CHECK: return %[[OUT1]] : memref<?x?xf32, "cpu">
  return %out1 : memref<?x?xf32, "cpu">
}
