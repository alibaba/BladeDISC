// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @sparse_fill_empty_rows
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?xi64>, %[[INPUT2:.*]]: memref<?xi64>, %[[INPUT3:.*]]: memref<?xi64>, %[[INPUT4:.*]]: memref<i64>, %[[OUT1:.*]]: memref<?x?xi64>, %[[OUT2:.*]]: memref<?xi64>, %[[OUT3:.*]]: memref<?xi1>, %[[OUT4:.*]]: memref<?xi64>, %[[OUT5:.*]]: memref<?xi64>) -> (memref<?x?xi64>, memref<?xi64>, memref<?xi1>, memref<?xi64>, memref<?xi64>)
func.func @sparse_fill_empty_rows(
    %input1: memref<?x?xi64>,
    %input2: memref<?xi64>,
    %input3: memref<?xi64>,
    %input4: memref<i64>,
    %out1: memref<?x?xi64>,
    %out2: memref<?xi64>,
    %out3: memref<?xi1>,
    %out4: memref<?xi64>,
    %out5: memref<?xi64>
    ) -> (
    memref<?x?xi64>,
    memref<?xi64>,
    memref<?xi1>,
    memref<?xi64>,
    memref<?xi64>
  ) {
  // CHECK-NOT lmhlo
  // CHECK:  %[[V4:.*]] = scf.for %[[ARG9:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG10:.*]] = %{{.*}}) -> (i64) {
  // CHECK:    %[[V14:.*]] = memref.load %{{.*}}[%[[ARG9]], %{{.*}}] : memref<?x?xi64>
  // CHECK:    %[[V15:.*]] = arith.index_cast %[[V14]] : i64 to index
  // CHECK:    %[[V16:.*]] = memref.load %{{.*}}[%[[V15]]] : memref<?xi1>
  // CHECK:    %[[V17:.*]] = scf.if %[[V16]] -> (i64) {
  // CHECK:      memref.store %{{.*}}, %{{.*}}[%[[V15]]] : memref<?xi1>
  // CHECK:      scf.yield %{{.*}} : i64
  // CHECK:    } else {
  // CHECK:      scf.yield %{{.*}} : i64
  // CHECK:    }
  // CHECK:    %[[V18:.*]] = arith.addi %{{.*}}, %[[V17]] : i64
  // CHECK:    %[[V19:.*]] = arith.index_cast %{{.*}} : index to i64
  // CHECK:    %[[V20:.*]] = arith.addi %[[V14]], %{{.*}} : i64
  // CHECK:    %[[V21:.*]] = arith.subi %[[V20]], %[[V18]] : i64
  // CHECK:    %[[V22:.*]] = arith.addi %[[V19]], %[[V21]] : i64
  // CHECK:    memref.store %[[V22]], %{{.*}}[%{{.*}}] : memref<?xi64>
  // CHECK:    %[[V23:.*]] = memref.load %{{.*}}[%[[V15]]] : memref<?xi64>
  // CHECK:    %[[V24:.*]] = arith.addi %[[V23]], %{{.*}} : i64
  // CHECK:    memref.store %[[V24]], %{{.*}}[%[[V15]]] : memref<?xi64>
  // CHECK:    scf.yield %[[V18]] : i64
  // CHECK:  }
  "lmhlo_disc.sparse_fill_empty_rows"(%input1, %input2, %input3, %input4, %out1, %out2, %out3, %out4, %out5) : (memref<?x?xi64>, memref<?xi64>, memref<?xi64>, memref<i64>, memref<?x?xi64>, memref<?xi64>, memref<?xi1>, memref<?xi64>, memref<?xi64>) -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]], %[[OUT3]], %[[OUT4]], %[[OUT5]] : memref<?x?xi64>, memref<?xi64>, memref<?xi1>, memref<?xi64>, memref<?xi64>
  return %out1, %out2, %out3, %out4, %out5 : memref<?x?xi64>, memref<?xi64>, memref<?xi1>, memref<?xi64>, memref<?xi64>
}
