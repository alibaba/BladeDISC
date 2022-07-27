// RUN: disc-opt -disc-memref-canonicalize %s -o - | FileCheck %s

// CHECK-LABEL: @no_alloc
func.func @no_alloc(%arg0: memref<?x?xf32, "gpu">, %arg1: index, %arg2: index) -> memref<?x?xf32, "gpu"> {
  // CHECK: memref.reinterpret_cast
  %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%arg1, %arg2], strides: [%arg2, 1] : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  return %0 : memref<?x?xf32, "gpu">
}

// CHECK-LABEL: @rank_not_match
func.func @rank_not_match(%arg0: memref<?x?xf32, "gpu">, %arg1: index) -> memref<?xf32, "gpu"> {
  // CHECK: memref.reinterpret_cast
  %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%arg1], strides: [1] : memref<?x?xf32, "gpu"> to memref<?xf32, "gpu">
  return %0 : memref<?xf32, "gpu">
}

// CHECK-LABEL: @should_convert
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @should_convert(%arg0: memref<?x?xf32, "gpu">, %arg1: index, %arg2: index) -> memref<?x?xf32, "gpu"> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // CHECK: %[[OUT:.*]] = memref.alloc(%[[ARG1]], %[[ARG2]])
  // CHECK-NOT: memref.reinterpret_cast
  // CHECK: return %[[OUT]]
  %0 = memref.alloc(%d0, %d1) : memref<?x?xf32, "gpu">
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [%arg1, %arg2], strides: [%arg2, 1] : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  return %1 : memref<?x?xf32, "gpu">
}


