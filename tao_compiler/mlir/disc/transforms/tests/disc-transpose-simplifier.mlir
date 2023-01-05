// RUN: disc-opt --disc-transpose-simplifier -split-input-file %s | FileCheck %s

// CHECK-LABEL: basic_test
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
func.func @basic_test(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
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
func.func @basic_test_2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
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
func.func @static_shape_test(%arg0: tensor<10x9xf32>) -> tensor<10x9xf32> {
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
func.func @const_operand(%arg0: tensor<10x9xf32>) -> tensor<10x9xf32> {
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
func.func @rand_3d_const_operand(%arg0: tensor<11x10x9xf32>) -> (tensor<11x10x9xf32>) {
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
func.func @should_not_convert_multi_output(%arg0: tensor<10x9xf32>) -> (tensor<9x10xf32>, tensor<10x9xf32>) {
  // CHECK: %[[T0:.*]] = mhlo.abs %[[ARG0]] : tensor<10x9xf32>
  // CHECK: "mhlo.transpose"(%[[T0]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  // CHECK-NOT: mhlo.transpose
  %cst = mhlo.constant dense<1.000000e+00> : tensor<9x10xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x10xf32>) -> tensor<9x10xf32>
  %2 = "mhlo.add"(%cst, %1) : (tensor<9x10xf32>, tensor<9x10xf32>) -> tensor<9x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  return %1, %3 : tensor<9x10xf32>, tensor<10x9xf32>
}

// -----

// CHECK-LABEL: should_not_convert_multi_output_2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x9xf32>) -> (tensor<9x10xf32>, tensor<9x10xf32>, tensor<10x9xf32>)
func.func @should_not_convert_multi_output_2(%arg0: tensor<10x9xf32>) -> (tensor<9x10xf32>, tensor<9x10xf32>, tensor<10x9xf32>) {
  // CHECK: "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  // CHECK: "mhlo.transpose"
  // CHECK-SAME: {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<9x10xf32>
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x9xf32>) -> tensor<9x10xf32>
  %1 = "mhlo.abs"(%0) : (tensor<9x10xf32>) -> tensor<9x10xf32>
  %2 = "mhlo.add"(%cst, %1) : (tensor<9x10xf32>, tensor<9x10xf32>) -> tensor<9x10xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<9x10xf32>) -> tensor<10x9xf32>
  return %0, %1, %3 : tensor<9x10xf32>, tensor<9x10xf32>, tensor<10x9xf32>
}

// -----

// CHECK-LABEL: shape_only_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
func.func @shape_only_consumer(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
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
func.func @reverse_transpose_test_0(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<3xindex>, %arg4: tensor<3xindex>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
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
func.func @reverse_transpose_test_1(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<3xindex>, %arg4: tensor<3xindex>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
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
func.func @reverse_transpose_test_2(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.add
  // CHECK: mhlo.transpose
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
}

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

// CHECK-LABEL: reverse_transpose_test_3
func.func @reverse_transpose_test_3(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: mhlo.transpose
  // CHECK: mhlo.add
  // CHECK: mhlo.transpose
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
}

// -----

// Check:
//  convert:
//                const ---
//                          \
//                           v
//   y -> transpose^{-1} -> add -> ... -> transpose -> yyy
//                              \
//                               -> ... -> zzz
//  to:
//                const' --
//                          \
//                           v
//                     y -> add -> ... -> yyy
//                              \
//                               -> transpose^{-1} ... -> zzz

// CHECK-LABEL: @reverse_transpose_test_3
func.func @reverse_transpose_test_3(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3xindex>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<3xindex>, %arg4: tensor<3xindex>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  // CHECK: mhlo.transpose
  // CHECK-NOT: mhlo.transpose
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %arg3) {broadcast_dimensions = dense<[]> : tensor<0xi64>, disc.device = "cpu"} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  %2 = "mhlo.transpose"(%arg2) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%2, %arg3) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  %4 = "mhlo.add"(%1, %3) {disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%4, %arg4) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 2, 1]> : tensor<3xi64>, disc.device = "cpu"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  return %4, %6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @move_up_transpsoe_greedily
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?x320xf32>, %[[ARG1:.*]]: tensor<?x?x?x320xf32>, %[[ARG2:.*]]: tensor<4xindex>, %[[ARG3:.*]]: tensor<4xindex>)
func.func @move_up_transpsoe_greedily(%arg0: tensor<?x?x?x320xf32>, %arg1: tensor<?x?x?x320xf32>, %arg2: tensor<4xindex>, %arg3: tensor<4xindex>) -> (tensor<?x320x?x?xf32>, tensor<?x?x?x320xf32> ) {
  // CHECK-DAG: %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<1x1x320xf32>
  // CHECK-DAG: %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[T2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[ARG2]]) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<1x1x320xf32>, tensor<4xindex>) -> tensor<?x?x?x320xf32>
  // CHECK: %[[T3:.*]] = mhlo.add %[[ARG1]], %[[T2]] : tensor<?x?x?x320xf32>

  // CHECK: %[[T4:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[ARG0]]
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1, 2, 3]>

  // CHECK: %[[T5:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[T3]]
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1, 2, 3]>

  // CHECK: %[[T6:.*]] = mhlo.add %[[T4]], %[[T5]] : tensor<?x?x?x320xf32>

  // CHECK: %[[T7:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[T1]]
  // CHECK-SAME: broadcast_dimensions = dense<>

  // CHECK: %[[T8:.*]] = mhlo.divide %6, %7 : tensor<?x?x?x320xf32>
  // CHECK: %[[T9:.*]] = "mhlo.transpose"(%8) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x?x?x320xf32>) -> tensor<?x320x?x?xf32>
  // CHECK: return %[[T9]], %[[T8]] : tensor<?x320x?x?xf32>, tensor<?x?x?x320xf32>

  %cst0 = mhlo.constant dense<1.0> : tensor<1x1x320xf32>
  %cst1 = mhlo.constant dense<0.0> : tensor<f32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%cst0, %arg2) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<1x1x320xf32>, tensor<4xindex>) -> tensor<?x?x?x320xf32>
  %1 = mhlo.add %arg1, %0 : tensor<?x?x?x320xf32>
  %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x?x?x320xf32>) -> tensor<?x320x?x?xf32>
  %3 = "mhlo.transpose"(%arg0) {disc.device = "gpu", permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x?x?x320xf32>) -> tensor<?x320x?x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%3, %arg3) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x320x?x?xf32>, tensor<4xindex>) -> tensor<?x320x?x?xf32>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%2, %arg3) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x320x?x?xf32>, tensor<4xindex>) -> tensor<?x320x?x?xf32>
  %6 = mhlo.add %4, %5 : tensor<?x320x?x?xf32>
  %7 = "mhlo.dynamic_broadcast_in_dim"(%cst1, %arg3) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "gpu"} : (tensor<f32>, tensor<4xindex>) -> tensor<?x320x?x?xf32>
  %8 = mhlo.divide %6, %7 : tensor<?x320x?x?xf32>
  %9 = "mhlo.transpose"(%8) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<?x320x?x?xf32>) -> tensor<?x?x?x320xf32>
  return %8, %9 : tensor<?x320x?x?xf32>, tensor<?x?x?x320xf32>
}

// -----

// CHECK-LABEL: @move_up_transpsoe_greedily_with_not_supported_ops
func.func @move_up_transpsoe_greedily_with_not_supported_ops(
    %arg0: tensor<?xf32>, %arg1: tensor<4xindex>,
    %arg2: tensor<?x?x?x?xf32>, %arg3: tensor<4xindex>,
    %arg4: tensor<?x?x?x?xf32>, %arg5: tensor<4xindex>) -> (tensor<?x?x?x?xf32>) {
  // only one transpose op should be left.
  // CHECK: mhlo.transpose
  // CHECK-NOT: mhlo.transpose
  %cst0 = mhlo.constant dense<1.0> : tensor<f32>
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %2 = "mhlo.transpose"(%arg2) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%cst0, %arg3) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %4 = mhlo.add %2, %3 : tensor<?x?x?x?xf32>

  %5 = "mhlo.transpose"(%arg4) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %6 = "mhlo.dynamic_broadcast_in_dim"(%cst0, %arg5) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %7 = mhlo.add %5, %6 : tensor<?x?x?x?xf32>

  %8 = mhlo.add %4, %7 : tensor<?x?x?x?xf32>
  %9 = mhlo.add %1, %8 : tensor<?x?x?x?xf32>

  %10 = "mhlo.transpose"(%9) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}
