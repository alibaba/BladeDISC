// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @where
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xf32, "cpu">, %[[OUT1:.*]]: memref<?x?xi64, "cpu">, %[[OUT2:.*]]: memref<1xi64, "cpu">) -> (memref<?x?xi64, "cpu">, memref<1xi64, "cpu">)
func.func @where(
    %input1: memref<?x?xf32, "cpu">,
    %out1: memref<?x?xi64, "cpu">,
    %out2: memref<1xi64, "cpu">
    ) -> (
    memref<?x?xi64, "cpu">,
    memref<1xi64, "cpu">
  ) {
  // CHECK-NOT lmhlo
  // CHECK: scf.for %[[ARG3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   %[[V1:.*]] = arith.index_cast %[[ARG3]] : index to i64
  // CHECK:   %[[V2:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32, "cpu">
  // CHECK:   scf.for %[[ARG4:.*]] = %{{.*}} to %[[V2]] step %{{.*}} {
  // CHECK:     %[[V3:.*]] = arith.index_cast %[[ARG4]] : index to i64
  // CHECK:     %[[V4:.*]] = memref.load %{{.*}}[%[[ARG3]], %[[ARG4]]] : memref<?x?xf32, "cpu">
  // CHECK:     %[[V5:.*]] = arith.cmpf one, %[[V4]], %{{.*}} : f32
  // CHECK:     scf.if %[[V5]] {
  // CHECK:        %[[V6:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<1xi64, "cpu">
  // CHECK:        %[[V7:.*]] = arith.index_cast %[[V6]] : i64 to index
  // CHECK:        memref.store %[[V1]], %{{.*}}[%[[V7]], %{{.*}}] : memref<?x?xi64, "cpu">
  // CHECK:        memref.store %[[V3]], %{{.*}}[%[[V7]], %{{.*}}] : memref<?x?xi64, "cpu">
  // CHECK:        %[[V8:.*]] = arith.addi %[[V6]], %{{.*}} : i64
  // CHECK:        memref.store %[[V8]], %{{.*}}[%{{.*}}] : memref<1xi64, "cpu">
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  "lmhlo_disc.where"(%input1, %out1, %out2) : (memref<?x?xf32, "cpu">, memref<?x?xi64, "cpu">, memref<1xi64, "cpu">) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?xi64, "cpu">, memref<1xi64, "cpu">
  return %out1, %out2 : memref<?x?xi64, "cpu">, memref<1xi64, "cpu">
}
