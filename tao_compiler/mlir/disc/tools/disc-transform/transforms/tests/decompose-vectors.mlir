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

// -----

// CHECK-LABEL: @test_if
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: vector<8xf32>, %[[ARG2:.*]]: vector<8xf32>)
func.func @test_if(%arg0: i1, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
  // CHECK: scf.if %[[ARG0]] -> (vector<4xf32>, vector<4xf32>)
  %out = scf.if %arg0 -> vector<8xf32> {
    // CHECK: %[[T0:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T1:.*]] = vector.extract_strided_slice %[[ARG2]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T2:.*]] = arith.addf %[[T0]], %[[T1]] : vector<4xf32>
    // CHECK: %[[T3:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T4:.*]] = vector.extract_strided_slice %[[ARG2]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T5:.*]] = arith.addf %[[T3]], %[[T4]] : vector<4xf32>
    // CHECK: scf.yield %[[T2]], %[[T5]] : vector<4xf32>, vector<4xf32>
    %0 = arith.addf %arg1, %arg2 : vector<8xf32>
    scf.yield %0 : vector<8xf32>
  } else {
    // CHECK: %[[T10:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T11:.*]] = vector.extract_strided_slice %[[ARG2]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T12:.*]] = arith.subf %[[T10]], %[[T11]] : vector<4xf32>
    // CHECK: %[[T13:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T14:.*]] = vector.extract_strided_slice %[[ARG2]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[T15:.*]] = arith.subf %[[T13]], %[[T14]] : vector<4xf32>
    // CHECK: scf.yield %[[T12]], %[[T15]] : vector<4xf32>, vector<4xf32>
    %1 = arith.subf %arg1, %arg2 : vector<8xf32>
    scf.yield %1 : vector<8xf32>
  }
  return %out: vector<8xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.disc.decompose_vectors %arg0 {vector_size = 4}
}
