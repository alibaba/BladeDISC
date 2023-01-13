// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_fma_xfer_ops
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>)
func.func @test_fma_xfer_ops(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG0]][%c0]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG0]][%c4]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG0]][%c8]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG0]][%c12]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c0]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c4]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c8]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c12]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG2]][%c0]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG2]][%c4]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG2]][%c8]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_read
  // CHECK-SAME: %[[ARG2]][%c12]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.fma
  // CHECK: vector.fma
  // CHECK: vector.fma
  // CHECK: vector.fma

  // CHECK: vector.transfer_write
  // CHECK-SAME: %[[ARG2]][%c0]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_write
  // CHECK-SAME: %[[ARG2]][%c4]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_write
  // CHECK-SAME: %[[ARG2]][%c8]
  // CHECK-SAME: vector<4xf32>

  // CHECK: vector.transfer_write
  // CHECK-SAME: %[[ARG2]][%c12]
  // CHECK-SAME: vector<4xf32>

  %0 = vector.transfer_read %arg0[%c0], %cst : memref<?xf32>, vector<16xf32>
  %1 = vector.transfer_read %arg1[%c0], %cst : memref<?xf32>, vector<16xf32>
  %2 = vector.transfer_read %arg2[%c0], %cst : memref<?xf32>, vector<16xf32>
  %3 = vector.fma %0, %1, %2 : vector<16xf32>
  vector.transfer_write %3, %arg2[%c0]: vector<16xf32>, memref<?xf32>
  return %arg2 : memref<?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.disc.decompose_vectors %arg0 {vector_size = 4}
}

// -----

// CHECK-LABEL: @test_for
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>)
func.func @test_for(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> memref<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %d0 = memref.dim %arg0, %c0 : memref<?xf32>
  // CHECK: %[[T0:.*]] = vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c0]
  // CHECK-SAME: vector<4xf32>

  // CHECK: %[[T1:.*]] = vector.transfer_read
  // CHECK-SAME: %[[ARG1]][%c4]
  // CHECK-SAME: vector<4xf32>

  // CHECK: %[[T2:.*]]:2 = scf.for %[[IV:.*]] = %c0 to %[[DIM0:.*]] step %c8
  // CHECK-SAME: iter_args(%[[ITER_ARG0:.*]] = %[[T0]], %[[ITER_ARG1:.*]] = %[[T1]]) -> (vector<4xf32>, vector<4xf32>)
  %0 = vector.transfer_read %arg1[%c0], %cst : memref<?xf32>, vector<8xf32>
  %3= scf.for %iv0 = %c0 to %d0 step %c8 iter_args(%arg2 = %0) -> (vector<8xf32>) {
      %1 = vector.transfer_read %arg0[%iv0], %cst : memref<?xf32>, vector<8xf32>
      // CHECK: arith.addf
      // CHECK-SAME: vector<4xf32>
      // CHECK: arith.addf
      // CHECK-SAME: vector<4xf32>
      %2 = arith.addf %1, %arg2 : vector<8xf32>
      scf.yield %2 : vector<8xf32>
  }
  // CHECK: vector.transfer_write
  // CHECK-SAME: vector<4xf32>
  // CHECK: vector.transfer_write
  // CHECK-SAME: vector<4xf32>
  vector.transfer_write %3, %arg1[%c0] : vector<8xf32>, memref<?xf32>
  return %arg1: memref<?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.disc.decompose_vectors %arg0 {vector_size = 4}
}
