// RUN: disc-opt --disc-memref-copy-to-linalg -split-input-file %s | FileCheck %s

// CHECK-LABEL: @copy
func.func @copy(%arg1: memref<?x?xf32, "cpu">, %arg2: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // CHECK-NOT: memref.copy
  // CHECK: linalg.generic
  memref.copy %arg1, %arg2 : memref<?x?xf32, "cpu"> to memref<?x?xf32, "cpu">
  return %arg2 : memref<?x?xf32, "cpu">
}