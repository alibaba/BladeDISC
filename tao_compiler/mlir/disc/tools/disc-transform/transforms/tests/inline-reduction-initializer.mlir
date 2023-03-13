// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @inline_reduction_loop_initializer
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?xf32>)
func.func @inline_reduction_loop_initializer(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) -> memref<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c372 = arith.constant 372 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  // CHECK: %[[NEW_INIT_BUFFER:.*]] = memref.alloca() {alignment = 64 : i64} : memref<16xf32>
  // CHECK-NEXT: linalg.fill ins(%cst : f32) outs(%[[NEW_INIT_BUFFER]] : memref<16xf32>)
  // CHECK-NOT: linalg.fill
  // CHECK: scf.for %[[IV0:.*]] = %c0 to %[[DIM0:.*]] step %c372
  // CHECK-NEXT:  %[[FIRST_ITER:.*]] = arith.cmpi eq, %[[IV0]], %c0 : index
  // CHECK-NEXT:  scf.for %[[IV1:.*]] = %c0 to %[[DIM1:.*]] step %c16
  // CHECK-NEXT:    %[[T0:.*]] = scf.if %[[FIRST_ITER]] -> (memref<?xf32>) {
  // CHECK-NEXT:      %[[CAST0:.*]] = memref.cast %[[NEW_INIT_BUFFER]] : memref<16xf32> to memref<?xf32>
  // CHECK-NEXT:      scf.yield %[[CAST0]] : memref<?xf32>
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      scf.yield %[[ARG1]] : memref<?xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %[[T1:.*]] = vector.transfer_read %[[T0]]
  linalg.fill ins(%cst : f32) outs(%arg1 : memref<?xf32>)
  scf.for %iv0 = %c0 to %d0 step %c372 {
    scf.for %iv1 = %c0 to %d1 step %c16 {
      %0 = vector.transfer_read %arg1[%iv1], %cst : memref<?xf32>, vector<16xf32>
      %1 = scf.for %iv2 = %c0 to %c372 step %c1 iter_args(%acc = %0) -> (vector<16xf32>) {
        %iv3 = arith.addi %iv0, %iv1 : index
        %1 = vector.transfer_read %arg0[%iv3, %iv1], %cst : memref<?x?xf32>, vector<16xf32>
        %2 = arith.addf %1, %acc : vector<16xf32>
        scf.yield %2 : vector<16xf32>
      }
      vector.transfer_write %1, %arg1[%iv1] : vector<16xf32>, memref<?xf32>
    }
  }

  return %arg0: memref<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %readers = transform.structured.match ops{["vector.transfer_read"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %reader_for_output, %reader_for_input = split_handles %readers in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %loop = transform.loop.get_parent_for %reader_for_output {num_loops = 2 : i64} : (!pdl.operation) -> !pdl.operation
  transform.disc.inline_reduction_initializer %fill for reader %reader_for_output into loop %loop
}
