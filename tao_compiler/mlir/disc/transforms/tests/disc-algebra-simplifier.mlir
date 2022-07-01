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

// -----

// CHECK-LABEL: @broadcast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x11xf32>) -> tensor<10x11xf32>
func @broadcast(%arg0 : tensor<10x11xf32>) -> tensor<10x11xf32> {
  // CHECK-NOT: mhlo.broadcast
  // CHECK: return %[[ARG0]] : tensor<10x11xf32>
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[]> : tensor<0xi64>} : (tensor<10x11xf32>) -> tensor<10x11xf32>
  return %0 : tensor<10x11xf32>
}

// -----

// CHECK-LABEL: @broadcast_in_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x11xf32>) -> tensor<10x11xf32>
func @broadcast_in_dim(%arg0 : tensor<10x11xf32>) -> tensor<10x11xf32> {
  // CHECK-NOT: mhlo.broadcast_in_dim
  // CHECK: return %[[ARG0]] : tensor<10x11xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<10x11xf32>) -> tensor<10x11xf32>
  return %0 : tensor<10x11xf32>
}

// -----

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@S0, @S1]>, %[[ARG1:.*]]: tensor<2xindex>) -> tensor<?x?xf32, [@S0, @S1]>
func @dynamic_broadcast_in_dim(%arg0 : tensor<?x?xf32, [@S0, @S1]>, %arg1: tensor<2xindex>) -> tensor<?x?xf32, [@S0, @S1]> {
  // CHECK-NOT: mhlo.dynamic_broadcast_in_dim
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32, [@S0, @S1]>, tensor<2xindex>) -> tensor<?x?xf32, [@S0, @S1]>
  // CHECK return %[[ARG0]] : tensor<?x?xf32, [@S0, @S1]>
  return %0 : tensor<?x?xf32, [@S0, @S1]>
}

// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "S0"
// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "S1"
"disc_shape.SymbolicDim"() {sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {sym_name = "S1", value = -1 : i64} : () -> ()

// -----

// CHECK-LABEL: broadcast_in_dim_of_reshape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x11xf32>, %[[ARG1:.*]]: tensor<10x11x12xf32>) -> tensor<10x11x12xf32>
func @broadcast_in_dim_of_reshape(%arg0: tensor<10x11xf32>, %arg1: tensor<10x11x12xf32>) -> tensor<10x11x12xf32> {
  // CHECK-NOT: mhlo.reshape
  // CHECK: %[[T0:.*]] = "mhlo.broadcast_in_dim"(%[[ARG0]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[ARG1]]
  // CHECK: return %[[T1]]
  %0 = "mhlo.reshape"(%arg0) : (tensor<10x11xf32>) -> tensor<10x11x1xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<10x11x1xf32>) -> tensor<10x11x12xf32>
  %2 = mhlo.add %1, %arg1 : tensor<10x11x12xf32>
  return %2 : tensor<10x11x12xf32>
}

// -----

// CHECK-LABEL: dynamic_broadcast_in_dim_of_reshape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@S0, @S1]>, %[[ARG1:.*]]: tensor<?x?x?xf32, [@S0, @S1, @S2]>) -> tensor<?x?x?xf32, [@S0, @S1, @S2]>
func @dynamic_broadcast_in_dim_of_reshape(%arg0: tensor<?x?xf32, [@S0, @S1]>, %arg1: tensor<?x?x?xf32, [@S0, @S1, @S2]>) -> tensor<?x?x?xf32, [@S0, @S1, @S2]> {
  // CHECK-NOT: mhlo.dynamic_reshape
  // CHECK: %[[T0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]]
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[ARG1]]
  // CHECK: return %[[T1]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32, [@S0, @S1]>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32, [@S0, @S1]>
  %2 = tensor.from_elements %0, %1, %c1 : tensor<3xindex>
  %3 = "mhlo.dynamic_reshape"(%arg0, %2) : (tensor<?x?xf32, [@S0, @S1]>, tensor<3xindex>) -> tensor<?x?x1xf32, [@S0, @S1, @C1]>
  %4 = tensor.dim %arg1, %c0 : tensor<?x?x?xf32, [@S0, @S1, @S2]>
  %5 = tensor.dim %arg1, %c1 : tensor<?x?x?xf32, [@S0, @S1, @S2]>
  %6 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32, [@S0, @S1, @S2]>
  %7 = tensor.from_elements %4, %5, %6 : tensor<3xindex>
  %8 = "mhlo.dynamic_broadcast_in_dim"(%3, %7) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x1xf32, [@S0, @S1, @C1]>, tensor<3xindex>) -> tensor<?x?x?xf32, [@S0, @S1, @S2]>
  %9 = mhlo.add %8, %arg1 : tensor<?x?x?xf32, [@S0, @S1, @S2]>
  return %9 : tensor<?x?x?xf32, [@S0, @S1, @S2]>
}

// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "S0"
// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "S1"
// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "S2"
// CHECK-DAG: "disc_shape.SymbolicDim"()
// CHECK-SAME: sym_name = "C1"
"disc_shape.SymbolicDim"() {sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {sym_name = "S1", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {sym_name = "S2", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {sym_name = "C1", value = 1 : i64} : () -> ()

// -----

// x + 0 = x

// CHECK-LABEL: @add_zero_tensor
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<2xindex>) -> tensor<?x?xf32>
func @add_zero_tensor(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: return %[[ARG0]] : tensor<?x?xf32>
  %0 = mhlo.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = mhlo.add %1, %arg0 : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----
// x * 1 = x

// CHECK-LABEL: @mul_one_tensor
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<2xindex>) -> tensor<?x?xf32>
func @mul_one_tensor(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: return %[[ARG0]] : tensor<?x?xf32>
  %0 = mhlo.constant dense<1.0> : tensor<f32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

