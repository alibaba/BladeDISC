// RUN: disc-opt %s -disc-convert-shape-to-std -split-input-file | FileCheck %s

// Lower `const_shape` to `tensor.from_elements`.
// CHECK-LABEL: @const_shape
// CHECK-SAME: () -> tensor<3xindex>
func @const_shape() -> tensor<3xindex> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[TENSOR3:.*]] = tensor.from_elements %[[C1]], %[[C2]], %[[C3]]
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[TENSOR3]] : tensor<3xindex> to tensor<3xindex>
  // CHECK: return %[[RESULT]] : tensor<3xindex>
  %shape = shape.const_shape [1, 2, 3] : tensor<3xindex>
  return %shape : tensor<3xindex>
}

// -----

// Lower `const_shape` in the case of rank 0.
// CHECK-LABEL: func @const_shape_zero_elements
// CHECK-SAME: () -> tensor<0xindex>
func @const_shape_zero_elements() -> tensor<0xindex> {
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements : tensor<0xindex>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[TENSOR]] : tensor<0xindex> to tensor<0xindex>
  // CHECK: return %[[RESULT]] : tensor<0xindex>
  %shape = shape.const_shape [] : tensor<0xindex>
  return %shape : tensor<0xindex>
}
