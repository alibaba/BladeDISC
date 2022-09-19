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
  // CHECK: scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   %[[V9:.*]] = memref.load %{{.*}}[%[[ARG4]]] : memref<?xf32, "cpu">
  // CHECK:   %[[V10:.*]] = arith.cmpf one, %[[V9]], %{{.*}} : f32
  // CHECK:   scf.if %[[V10]] {
  // CHECK:     scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:       %[[V11:.*]] = memref.load %{{.*}}[%[[ARG4]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:       %[[V12:.*]] = arith.divf %[[V11]], %[[V9]] : f32
  // CHECK:       memref.store %[[V12]], %{{.*}}[%[[ARG4]], %[[ARG5]]] : memref<?x?xf32, "cpu">
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  "lmhlo_disc.sparse_segment_mean"(%input1, %input2, %input3, %out1) : (memref<?x?xf32, "cpu">, memref<?xi32, "cpu">, memref<?xi32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  // CHECK: return %[[OUT1]] : memref<?x?xf32, "cpu">
  return %out1 : memref<?x?xf32, "cpu">
}
