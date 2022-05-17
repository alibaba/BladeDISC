// RUN: disc-opt --disc-shape-optimization -split-input-file %s | FileCheck %s

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>) -> tensor<10xf32>
func @main(%arg0 : tensor<?xf32>) -> tensor<10xf32> {
  // CHECK: return %[[ARG0]] : tensor<10xf32>
  %0 = tensor.cast %arg0 : tensor<?xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

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
  // CHECK: return %[[T1]] : tensor<?x20xf32, [@[[S0]], @[[S2]]]>
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 1 : i64 } : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
