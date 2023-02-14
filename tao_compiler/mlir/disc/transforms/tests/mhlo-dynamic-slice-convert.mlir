// RUN: disc-opt -disc-dynamic-slice-converter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<?xi64>, %arg1: tensor<i32>) -> tensor<1xi64> {
  // CHECK: %[[V0:.*]] = tensor.dim %[[ARG0:.*]], %{{.*}} : tensor<?xi64>
  // CHECK: %[[V1:.*]] = arith.index_cast %[[V0]] : index to i64
  // CHECK: %[[V2:.*]] = arith.subi %[[V1]], %{{.*}} : i64
  // CHECK: %[[V3:.*]] = tensor.extract %[[ARG1:.*]][] : tensor<i32>
  // CHECK: %[[V4:.*]] = arith.extsi %[[V3]] : i32 to i64
  // CHECK: %[[V5:.*]] = arith.maxsi %[[V4]], %{{.*}} : i64
  // CHECK: %[[V6:.*]] = arith.minsi %[[V5]], %[[V2]] : i64
  // CHECK: %[[V7:.*]] = arith.addi %[[V6]], %{{.*}} : i64
  // CHECK: %[[V8:.*]] = tensor.from_elements %[[V6]] : tensor<1xi64>
  // CHECK: %[[V9:.*]] = tensor.from_elements %[[V7]] : tensor<1xi64>
  // CHECK: %[[V10:.*]] = mhlo.real_dynamic_slice %[[ARG0:.*]], %[[V8]], %[[V9]], %{{.*}} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %out = "mhlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<?xi64>, tensor<i32>) -> tensor<1xi64>
  return %out : tensor<1xi64>
}

// -----

// CHECK-LABEL: func.func @dynamic_slice_with_constant_start_indices
func.func @dynamic_slice_with_constant_start_indices(%arg0: tensor<?xi64>) -> tensor<1xi64> {
  // CHECK: %[[CST1:.*]] = arith.constant dense<1> : tensor<1xi64>
  // CHECK: %[[CST0:.*]] = arith.constant dense<0> : tensor<1xi64>
  // CHECK: mhlo.real_dynamic_slice %[[ARG0:.*]], %[[CST0]], %[[CST1]], %[[CST1]] : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %start_indices = mhlo.constant dense<0> : tensor<i32>
  %out = "mhlo.dynamic_slice"(%arg0, %start_indices) {
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<?xi64>, tensor<i32>) -> tensor<1xi64>
  return %out : tensor<1xi64>
}
