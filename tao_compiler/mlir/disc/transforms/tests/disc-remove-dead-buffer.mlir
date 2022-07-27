// RUN: disc-opt -canonicalize --disc-remove-dead-buffer -split-input-file %s | FileCheck %s

// CHECK-LABEL: not_removable_basic_test
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>) -> memref<?xf32>
func.func @not_removable_basic_test(%arg0 : memref<?xf32>) -> memref<?xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}}  {
  // CHECK: memref.alloc
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32>
  %1 = memref.alloc(%0) : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// CHECK-LABEL: removable_basic_test_0
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>) -> memref<?xf32>
func.func @removable_basic_test_0(%arg0 : memref<?xf32>) -> memref<?xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}}  {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.cast
  // CHECK-NOT: memref.dealloc
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32>
  %1 = memref.alloc(%0) : memref<?xf32>
  %2 = memref.cast %1 : memref<?xf32> to memref<4xf32>
  memref.dealloc %2 : memref<4xf32>
  return %arg0 : memref<?xf32>
}
