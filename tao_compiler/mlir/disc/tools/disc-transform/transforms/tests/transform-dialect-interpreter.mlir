// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s --check-prefix=INLINE
// RUN: disc-opt --disc-transform-dialect-interpreter=transform-file-name=%p/metadata-only-transform-dialect-interpreter-standalone.mlir -split-input-file %s | FileCheck %s --check-prefix=STANDALONE

// INLINE-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  // INLINE: %{{.*}} = scf.for
  // INLINE-NEXT: %{{.*}} = scf.for
  // INLINE-NEXT: %{{.*}} = scf.for
  // INLINE: tensor.extract_slice
  // INLINE-NEXT: tensor.extract_slice
  // INLINE-NEXT: tensor.extract_slice
  // INLINE-NEXT: linalg.matmul
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}


transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %loops:3 = transform.structured.tile %0 [2, 3, 4]
}

// -----

// STANDALONE-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  // STANDALONE: %{{.*}} = scf.for
  // STANDALONE-NEXT: %{{.*}} = scf.for
  // STANDALONE-NEXT: %{{.*}} = scf.for
  // STANDALONE: tensor.extract_slice
  // STANDALONE-NEXT: tensor.extract_slice
  // STANDALONE-NEXT: tensor.extract_slice
  // STANDALONE-NEXT: linalg.matmul
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}