// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @where
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xf32>, %[[OUT1:.*]]: memref<?x?xi64>, %[[OUT2:.*]]: memref<1xi64>) -> (memref<?x?xi64>, memref<1xi64>)
func.func @where(
    %input1: memref<?x?xf32>,
    %out1: memref<?x?xi64>,
    %out2: memref<1xi64>
    ) -> (
    memref<?x?xi64>,
    memref<1xi64>
  ) {
  // CHECK-NOT lmhlo
  // CHECK: %[[D:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
  // CHECK: %[[D_0:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?x?xf32>
  // CHECK: scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%{{.*}}, %{{.*}}) to (%[[D]], %[[D_0]]) step (%{{.*}}, %{{.*}}) {
  // CHECK:   %[[V0:.*]] = arith.index_cast %[[ARG3]] : index to i64
  // CHECK:   %[[V1:.*]] = arith.index_cast %[[ARG4]] : index to i64
  // CHECK:   %[[V2:.*]] = memref.load %{{.*}}[%[[ARG3]], %[[ARG4]]] : memref<?x?xf32>
  // CHECK:   %[[V3:.*]] = arith.cmpf one, %[[V2]], %{{.*}} : f32
  // CHECK:   scf.if %[[V3]] {
  // CHECK:     %[[V4:.*]] = memref.load %{{.*}}[] : memref<i64>
  // CHECK:     %[[V5:.*]] = arith.index_cast %[[V4]] : i64 to index
  // CHECK:     memref.store %[[V0]], %{{.*}}[%[[V5]], %{{.*}}] : memref<?x?xi64>
  // CHECK:     memref.store %[[V1]], %{{.*}}[%[[V5]], %{{.*}}] : memref<?x?xi64>
  // CHECK:     %[[V6:.*]] = arith.addi %[[V4]], %{{.*}} : i64
  // CHECK:     memref.store %[[V6]], %{{.*}}[] : memref<i64>
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }
  "lmhlo_disc.where"(%input1, %out1, %out2) : (memref<?x?xf32>, memref<?x?xi64>, memref<1xi64>) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?xi64>, memref<1xi64>
  return %out1, %out2 : memref<?x?xi64>, memref<1xi64>
}

// -----

// CHECK-LABEL: @where_fusion
// CHECK-SAME: (%[[INPUT1:.*]]: memref<i64>, %[[INPUT2:.*]]: memref<3xindex>, %[[INPUT3:.*]]: memref<?x?x?xi64>, %[[INPUT4:.*]]: memref<?x?x?xi64>, %[[INPUT5:.*]]: memref<?x?x?xi1>, %[[OUT1:.*]]: memref<?x3xi64>, %[[OUT2:.*]]: memref<1xi64>) -> (memref<?x3xi64>, memref<1xi64>)
func.func @where_fusion(
    %input1: memref<i64>,
    %input2: memref<3xindex>,
    %input3: memref<?x?x?xi64>,
    %input4: memref<?x?x?xi64>,
    %input5: memref<?x?x?xi1>,
    %out1: memref<?x3xi64>,
    %out2: memref<1xi64>
    ) -> (
    memref<?x3xi64>,
    memref<1xi64>
  ) {
  // CHECK lmhlo.fusion
  // CHECK lmhlo.constant
  // CHECK lmhlo.dynamic_broadcast_in_dim
  // CHECK lmhlo.compare
  // CHECK-NOT lmhlo.where
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%input1) {disc.device = "cpu", value = dense<0> : tensor<i64>} : (memref<i64>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %input3) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i64>, memref<3xindex>, memref<?x?x?xi64>) -> ()
    "lmhlo.compare"(%input4, %input3, %input5) {comparison_direction = #mhlo<comparison_direction NE>, disc.device = "cpu"} : (memref<?x?x?xi64>, memref<?x?x?xi64>, memref<?x?x?xi1>) -> ()
    "lmhlo_disc.where"(%input5, %out1, %out2) {disc.device = "cpu"} : (memref<?x?x?xi1>, memref<?x3xi64>, memref<1xi64>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "main_kWhere_where__4_2_0", disc.fusion_type = "kWhere"} : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x3xi64>, memref<1xi64>
  return %out1, %out2 : memref<?x3xi64>, memref<1xi64>
}
