// RUN: disc-opt --disc-convert-tensor-to-std -split-input-file %s | FileCheck %s

// CHECK-LABEL: generate_op
func.func @generate_op(%arg0: tensor<?x?xf32>) -> tensor<2xi32> attributes {tf.entry_function = {inputs = "input0", outputs = "output0"}} {
  // CHECK: %[[T0:.*]] = shape.shape_of {{.*}} : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[T1:.*]] = tensor.extract %[[T0]][%c0] : tensor<2xindex>
  // CHECK: %[[T2:.*]] = arith.index_cast %[[T1]] : index to i32
  // CHECK: %[[T3:.*]] = tensor.extract %0[%c1] : tensor<2xindex>
  // CHECK: %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
  // CHECK: %[[T5:.*]] = tensor.from_elements %[[T2]], %[[T4]] : tensor<2xi32>
  // CHECK: return %[[T5]] : tensor<2xi32>
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %1 = tensor.generate   {
  ^bb0(%arg1: index):  // no predecessors
    %2 = tensor.extract %0[%arg1] : tensor<2xindex>
    %3 = arith.index_cast %2 : index to i32
    tensor.yield %3 : i32
  } : tensor<2xi32>
  return %1 : tensor<2xi32>
}
