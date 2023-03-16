// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s --check-prefix=INLINE
// RUN: disc-opt --disc-transform-dialect-interpreter=transform-file-name=%p/metadata-only-transform-dialect-interpreter-standalone.mlir -split-input-file %s | FileCheck %s --check-prefix=STANDALONE

// INLINE-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  // INLINE: %{{.*}} = scf.for
  // INLINE: %{{.*}} = scf.for
  // INLINE: %{{.*}} = scf.for
  // INLINE: tensor.extract_slice
  // INLINE-NEXT: tensor.extract_slice
  // INLINE-NEXT: tensor.extract_slice
  // INLINE-NEXT: linalg.matmul
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}


transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:3 = transform.structured.tile %0 [2, 3, 4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
}

// -----

// INLINE-LABEL: @matmul_nn
// INLINE-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>)
func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // INLINE-CHECK: scf.foreach_thread
  // INLINE: %[[SUBVIEW0:.*]] = memref.subview %[[ARG0]]
  // INLINE-NEXT: %[[SUBVIEW1:.*]] = memref.subview %[[ARG1]]
  // INLINE-NEXT: %[[SUBVIEW2:.*]] = memref.subview %[[ARG2]]
  // INLINE-NEXT: linalg.fill ins(%cst : f32) outs(%[[SUBVIEW2]]
  // INLINE-NEXT: linalg.matmul ins(%[[SUBVIEW0]], %[[SUBVIEW1]]
  // INLINE-SAEM: outs(%[[SUBVIEW2]]
  // INLINE: return %[[ARG2]]
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation

  %0:2 = transform.structured.tile_to_foreach_thread_op %matmul tile_sizes [6, 16]
  transform.structured.fuse_into_containing_op %fill into %0#0

  transform.disc.bufferize %arg1
}

// -----

// STANDALONE-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  // STANDALONE: %{{.*}} = scf.for
  // STANDALONE: %{{.*}} = scf.for
  // STANDALONE: %{{.*}} = scf.for
  // STANDALONE: tensor.extract_slice
  // STANDALONE-NEXT: tensor.extract_slice
  // STANDALONE-NEXT: tensor.extract_slice
  // STANDALONE-NEXT: linalg.matmul
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}