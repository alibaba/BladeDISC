// RUN: disc-opt --disc-memref-copy-to-linalg -split-input-file %s | FileCheck %s

// CHECK-LABEL: @copy
func.func @copy(%arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> memref<?x?xf32> {
  // CHECK-NOT: memref.copy
  // CHECK: linalg.generic
  memref.copy %arg1, %arg2 : memref<?x?xf32> to memref<?x?xf32>
  return %arg2 : memref<?x?xf32>
}