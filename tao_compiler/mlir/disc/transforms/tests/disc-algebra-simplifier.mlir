// RUN: disc-opt -disc-algebra-simplifier -canonicalize -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: splat_const_integer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x11xf32>)
func @splat_const_integer(%arg0 : tensor<10x11xf32>) -> tensor<10x11xf32> {
  // CHECK-NOT: mhlo.power
  // CHECK: %[[T0:.*]] = mhlo.multiply %[[ARG0]], %[[ARG0]]
  // CHECK: %[[T1:.*]] = mhlo.multiply %[[T0]], %[[ARG0]]
  // CHECK: return %[[T1]]
  %0 = mhlo.constant dense<3.0e+00> : tensor<10x11xf32>
  %1 = mhlo.power %arg0, %0 : tensor<10x11xf32>
  return %1 : tensor<10x11xf32>
}

// -----

// CHECK-LABEL: splat_const_not_integer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x11xf32>)
func @splat_const_not_integer(%arg0 : tensor<10x11xf32>) -> tensor<10x11xf32> {
  // CHECK-NOT: mhlo.multiply
  // CHECK: %[[T0:.*]] = mhlo.power
  // CHECK: return %[[T0]]
  %0 = mhlo.constant dense<3.3e+00> : tensor<10x11xf32>
  %1 = mhlo.power %arg0, %0 : tensor<10x11xf32>
  return %1 : tensor<10x11xf32>
}

// -----

// CHECK-LABEL: bcast_const_integer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>)
func @bcast_const_integer(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = mhlo.multiply %[[ARG0]], %[[ARG0]]
  // CHECK: %[[T1:.*]] = mhlo.multiply %[[T0]], %[[ARG0]]
  // CHECK: return %[[T1]]
  %0 = mhlo.constant dense<3.0e+00> : tensor<f32>
  %1 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %3 = mhlo.power %arg0, %2 : tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: bcast_const_not_integer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>)
func @bcast_const_not_integer(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: mhlo.multiply
  // CHECK: %[[T0:.*]] = mhlo.power
  // CHECK: return %[[T0]]
  %0 = mhlo.constant dense<3.01e+00> : tensor<f32>
  %1 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %3 = mhlo.power %arg0, %2 : tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}


