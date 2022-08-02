// RUN: disc-opt -disc-flatten-memref-access %s -o - | FileCheck %s

// CHECK-LABEL: @load_not_convert
func.func @load_not_convert(%arg0: memref<?x?xf32, "gpu">) -> f32 {
  // CHECK-NOT: disc_shape.linearize
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.load %arg0[%c0, %c1] : memref<?x?xf32, "gpu">
  return %0 : f32
}

// CHECK-LABEL: @load_convert
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">)
func.func @load_convert(%arg0: memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK scf.parallel
  scf.parallel (%arg1) = (%c0) to (%c1) step (%c1) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG0]], %[[C0]]
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[DIM1:.*]] = memref.dim %[[ARG0]], %[[C1]]
    // CHECK: %[[LINEAR_IDX:.*]] = "disc_shape.linearize"(%c0, %c1, %[[DIM0]], %[[DIM1]])

    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0]]
    // CHECK-SAME: memref<?x?xf32, "gpu"> to memref<?xf32, "gpu">
    // CHECK: %[[LOAD:.*]] = memref.load %[[CAST]]
    %0 = memref.load %arg0[%c0, %c1] : memref<?x?xf32, "gpu">
  }
  return
}

// CHECK-LABEL: @store_not_convert
func.func @store_not_convert(%arg0: memref<?x?xf32, "gpu">, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %arg1,  %arg0[%c0, %c1] : memref<?x?xf32, "gpu">
  return
}

// CHECK-LABEL: @store_convert
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: f32)
func.func @store_convert(%arg0: memref<?x?xf32, "gpu">, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK scf.parallel
  scf.parallel (%arg2) = (%c0) to (%c1) step (%c1) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG0]], %[[C0]]
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[DIM1:.*]] = memref.dim %[[ARG0]], %[[C1]]
    // CHECK: %[[LINEAR_IDX:.*]] = "disc_shape.linearize"(%c0, %c1, %[[DIM0]], %[[DIM1]])

    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0]]
    // CHECK-SAME: memref<?x?xf32, "gpu"> to memref<?xf32, "gpu">
    // CHECK: memref.store %[[ARG1]], %[[CAST]]
    memref.store %arg1, %arg0[%c0, %c1] : memref<?x?xf32, "gpu">
  }
  return
}
