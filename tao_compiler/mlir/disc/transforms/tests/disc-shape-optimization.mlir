// RUN: disc-opt --disc-shape-optimization -split-input-file %s | FileCheck %s

// Test tensor.cast op

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>) -> tensor<10xf32>
func @main(%arg0 : tensor<?xf32>) -> tensor<10xf32> {
  // CHECK: return %[[ARG0]] : tensor<10xf32>
  %0 = tensor.cast %arg0 : tensor<?xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// Test mhlo.add op

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>) -> tensor<10xf32>
func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<10xf32>) -> tensor<?xf32> {
  %0 = tensor.cast %arg1 : tensor<10xf32> to tensor<?xf32>
  // CHECK: %[[T1:.*]] = mhlo.add %[[ARG0]], %[[ARG1]] : tensor<10xf32>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: return %[[T1]] : tensor<10xf32>
  return %1 : tensor<?xf32>
}

// -----

// Test mhlo.concat op
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x10xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<?x10xf32, [@[[S0]], @[[S1]]]>) -> tensor<?x20xf32, [@[[S0]], @[[S2:.*]]]>
func @main(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.concatenate"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x20xf32, [@[[S0]], @[[S2]]]>
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 1 : i64 } : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test mhlo.dot_general
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]], @[[S3:.*]]]>, %[[ARG1:.*]]: tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]], @[[S4:.*]]]>) -> tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S2]], @[[S4]]]>
func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S2]], @[[S4]]]>
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// Test mhlo.dot
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<?x?xf32, [@[[S1]], @[[S2:.*]]]>) -> tensor<?x?xf32, [@[[S0]], @[[S2]]]>
func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?xf32, [@[[S0]], @[[S2]]]>
  %1 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

// Test mhlo.clamp: zero rank min/max

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.clamp"(%[[ARG1]], %[[ARG0]], %[[ARG2]])
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<f32>, tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.clamp: same shape

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.clamp"(%[[ARG1]], %[[ARG0]], %[[ARG2]])
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.select : zero rank
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0: tensor<i1>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.select"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2)  : (tensor<i1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.select : same shape
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xi1, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0: tensor<?xi1>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.select"(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2)  : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.einsum
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]]]>, %[[ARG1:.*]]: tensor<?x?x?xf32, [@[[S0]], @[[S2]], @[[S3:.*]]]>) -> tensor<?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]]]>
func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "input0,input1", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: %[[T0:.*]] = "mhlo.einsum"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]]]>
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
