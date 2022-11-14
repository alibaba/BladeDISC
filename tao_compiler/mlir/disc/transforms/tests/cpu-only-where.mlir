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
  // CHECK: %[[D:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32, "cpu">
  // CHECK: %[[D_0:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32, "cpu">
  // CHECK: scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%{{.*}}, %{{.*}}) to (%[[D]], %[[D_0]]) step (%{{.*}}, %{{.*}}) {
  // CHECK:   %[[V0:.*]] = arith.index_cast %[[ARG3]] : index to i64
  // CHECK:   %[[V1:.*]] = arith.index_cast %[[ARG4]] : index to i64
  // CHECK:   %[[V2:.*]] = memref.load %{{.*}}[%[[ARG3]], %[[ARG4]]] : memref<?x?xf32, "cpu">
  // CHECK:   %[[V3:.*]] = arith.cmpf one, %[[V2]], %{{.*}} : f32
  // CHECK:   scf.if %[[V3]] {
  // CHECK:     %[[V4:.*]] = memref.load %{{.*}}[] : memref<i64, "cpu">
  // CHECK:     %[[V5:.*]] = arith.index_cast %[[V4]] : i64 to index
  // CHECK:     memref.store %[[V0]], %{{.*}}[%[[V5]], %{{.*}}] : memref<?x?xi64, "cpu">
  // CHECK:     memref.store %[[V1]], %{{.*}}[%[[V5]], %{{.*}}] : memref<?x?xi64, "cpu">
  // CHECK:     %[[V6:.*]] = arith.addi %[[V4]], %{{.*}} : i64
  // CHECK:     memref.store %[[V6]], %{{.*}}[] : memref<i64, "cpu">
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }
  "lmhlo_disc.where"(%input1, %out1, %out2) : (memref<?x?xf32, "cpu">, memref<?x?xi64, "cpu">, memref<1xi64, "cpu">) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?xi64, "cpu">, memref<1xi64, "cpu">
  return %out1, %out2 : memref<?x?xi64, "cpu">, memref<1xi64, "cpu">
}

// -----

// CHECK-LABEL: @where_fusion
// CHECK-SAME: (%[[INPUT1:.*]]: memref<i64, "cpu">, %[[INPUT2:.*]]: memref<3xindex, "cpu">, %[[INPUT3:.*]]: memref<?x?x?xi64, "cpu">, %[[INPUT4:.*]]: memref<?x?x?xi64, "cpu">, %[[INPUT5:.*]]: memref<?x?x?xi1, "cpu">, %[[OUT1:.*]]: memref<?x3xi64, "cpu">, %[[OUT2:.*]]: memref<1xi64, "cpu">) -> (memref<?x3xi64, "cpu">, memref<1xi64, "cpu">)
func.func @where_fusion(
    %input1: memref<i64, "cpu">,
    %input2: memref<3xindex, "cpu">,
    %input3: memref<?x?x?xi64, "cpu">,
    %input4: memref<?x?x?xi64, "cpu">,
    %input5: memref<?x?x?xi1, "cpu">,
    %out1: memref<?x3xi64, "cpu">,
    %out2: memref<1xi64, "cpu">
    ) -> (
    memref<?x3xi64, "cpu">,
    memref<1xi64, "cpu">
  ) {
  // CHECK lmhlo.fusion
  // CHECK lmhlo.constant
  // CHECK lmhlo.dynamic_broadcast_in_dim
  // CHECK lmhlo.compare
  // CHECK-NOT lmhlo.where
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%input1) {disc.device = "cpu", value = dense<0> : tensor<i64>} : (memref<i64, "cpu">) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %input3) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i64, "cpu">, memref<3xindex, "cpu">, memref<?x?x?xi64, "cpu">) -> ()
    "lmhlo.compare"(%input4, %input3, %input5) {comparison_direction = #mhlo<comparison_direction NE>, disc.device = "cpu"} : (memref<?x?x?xi64, "cpu">, memref<?x?x?xi64, "cpu">, memref<?x?x?xi1, "cpu">) -> ()
    "lmhlo_disc.where"(%input5, %out1, %out2) {disc.device = "cpu"} : (memref<?x?x?xi1, "cpu">, memref<?x3xi64, "cpu">, memref<1xi64, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "main_kWhere_where__4_2_0", disc.fusion_type = "kWhere"} : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x3xi64, "cpu">, memref<1xi64, "cpu">
  return %out1, %out2 : memref<?x3xi64, "cpu">, memref<1xi64, "cpu">
}
