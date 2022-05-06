// RUN: disc-opt --disc-transpose-simplifier -split-input-file %s | FileCheck %s

// CHECK-LABEL: basic_test
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
func @basic_test(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[RET:.*]] = mhlo.abs %[[ARG0]]
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RET]]
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.transpose"(%1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: basic_test_2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
func @basic_test_2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[T0:.*]] = mhlo.abs %[[ARG0]]
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[RET:.*]] = mhlo.add %[[ARG0]], %[[T0]]
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RET]]
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: static_shape_test
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x9xf32>) -> tensor<10x9xf32>
func @static_shape_test(%arg0: tensor<10x9xf32>) -> tensor<10x9xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[T0:.*]] = mhlo.abs %[[ARG0]] : tensor<10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[RET:.*]] = mhlo.add %[[ARG0]], %[[T0]] : tensor<10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RET]] : tensor<10x9xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x10xf32>) -> tensor<9x10xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<9x10xf32>, tensor<9x10xf32>) -> tensor<9x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  return %3 : tensor<10x9xf32>
}

// -----

// CHECK-LABEL: const_operand
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x9xf32>) -> tensor<10x9xf32>
func @const_operand(%arg0: tensor<10x9xf32>) -> tensor<10x9xf32> {
  // CHECK: %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[T0:.*]] = mhlo.abs %[[ARG0]] : tensor<10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[RET:.*]] = mhlo.subtract %[[CST]], %[[T0]] : tensor<10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RET]] : tensor<10x9xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<9x10xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x10xf32>) -> tensor<9x10xf32>
  %2 = "mhlo.subtract"(%cst, %1) : (tensor<9x10xf32>, tensor<9x10xf32>) -> tensor<9x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  return %3 : tensor<10x9xf32>
}

// -----

// CHECK-LABEL: rand_3d_const_operand
// CHECK-SAME: (%[[ARG0:.*]]: tensor<11x10x9xf32>) -> tensor<11x10x9xf32>
func @rand_3d_const_operand(%arg0: tensor<11x10x9xf32>) -> (tensor<11x10x9xf32>) {
  // CHECK: %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<11x10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[T0:.*]] = mhlo.abs %[[ARG0]] : tensor<11x10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[RET:.*]] = mhlo.subtract %[[T0]], %[[CST]] : tensor<11x10x9xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RET]] : tensor<11x10x9xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<9x11x10xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[2, 0, 1]> : tensor<3xi64>} : (tensor<11x10x9xf32>) -> tensor<9x11x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x11x10xf32>) -> tensor<9x11x10xf32>
  %2 = "mhlo.subtract"(%1, %cst) : (tensor<9x11x10xf32>, tensor<9x11x10xf32>) -> tensor<9x11x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<9x11x10xf32>) -> tensor<11x10x9xf32>
  return %3 : tensor<11x10x9xf32>
}

// -----

// CHECK-LABEL: should_not_convert_multi_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x9xf32>) -> (tensor<9x10xf32>, tensor<10x9xf32>)
func @should_not_convert_multi_output(%arg0: tensor<10x9xf32>) -> (tensor<9x10xf32>, tensor<10x9xf32>) {
  // CHECK: "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  // CHECK: "mhlo.transpose"
  // CHECK-SAME: {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<9x10xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x10xf32>) -> tensor<9x10xf32>
  %2 = "mhlo.add"(%cst, %1) : (tensor<9x10xf32>, tensor<9x10xf32>) -> tensor<9x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  return %1, %3 : tensor<9x10xf32>, tensor<10x9xf32>
}

// -----

// CHECK-LABEL: shape_only_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
func @shape_only_consumer(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  // CHECK: %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?x?xf32>
  // CHECK: %[[D2:.*]] = tensor.dim %[[ARG0]], %c2 : tensor<?x?x?xf32>
  // CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %c1 : tensor<?x?x?xf32>
  // CHECK: %[[T1:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]] : tensor<3xindex>
  // CHECK: %[[T2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CST]], %[[T1]]) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  // CHECK: %[[T3:.*]] = mhlo.add %[[ARG0]], %[[T2]] : tensor<?x?x?xf32>
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[T3]] : tensor<?x?x?xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?x?xf32> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%cst, %1) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %3 = "mhlo.add"(%0, %2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "mhlo.transpose"(%3) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %4 : tensor<?x?x?xf32>
}

// -----

// Check:
//  convert:
//                      -> ... -> transpose -> xxx
//                     /
//                    x ---
//                          \
//                           v
//   y -> transpose^{-1} -> add -> ... -> transpose -> yyy
//                              \
//                               -> ... -> zzz
//  to:
//                        -> ... -> xxx
//                       /
//     x -> transpose(1, 0)
//                          \
//                           v
//                     y -> add -> ... -> yyy
//                              \
//                               -> transpose^{-1} ... -> zzz

// CHECK-LABEL: reverse_transpose_test_0
func @reverse_transpose_test_0(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<3xindex>, %arg4: tensor<3xindex>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  // CHECK: mhlo.transpose
  // CHECK: mhlo.transpose
  // CHECK-NOT: mhlo.transpose
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %2 = "mhlo.transpose"(%arg2) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%2, %arg3) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  %4 = "mhlo.add"(%arg0, %3) {disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%4, %arg4) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  return %1, %4, %6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: reverse_transpose_test_1
func @reverse_transpose_test_1(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<3xindex>, %arg4: tensor<3xindex>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  // CHECK: mhlo.transpose
  // CHECK: mhlo.transpose
  // CHECK-NOT: mhlo.transpose
  %cst = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %2 = "mhlo.transpose"(%arg2) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %21 = "mhlo.dynamic_broadcast_in_dim"(%cst, %arg3) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %22 = "mhlo.add"(%2, %21) {disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%22, %arg3) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  %4 = "mhlo.add"(%arg0, %3) {disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%4, %arg4) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  return %1, %4, %6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

// Check:
//  convert:
//   x -> transpose ---
//                      \
//                       v
//   y -> transpose --> add --> ...
//  to:
//   x ---
//        \
//         v
//   y --> add -> transpose -> ...

// CHECK-LABEL: reverse_transpose_test_2
func @reverse_transpose_test_2(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.add
  // CHECK: mhlo.transpose
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
}
