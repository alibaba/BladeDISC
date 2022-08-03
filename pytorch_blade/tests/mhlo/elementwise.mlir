// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.sigmoid(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = mhlo.negate %[[ARG0]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = mhlo.exponential %[[T1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.add %[[T2]], %[[T0]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.divide %[[T0]], %[[T3]] : tensor<?x?xf32>
// CHECK:         return %[[T4]] : tensor<?x?xf32>
func.func @torch.aten.sigmoid(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func @torch.aten.relu(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 0.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = "mhlo.compare"(%[[ARG0]], %[[T0]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:         %[[T2:.*]] = "mhlo.select"(%[[T1]], %[[ARG0]], %[[T0]]) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T2]] : tensor<?x?xf32>
func.func @torch.aten.relu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func @torch.aten.gelu(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 2.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 5.000000e-01 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.rsqrt %[[T1]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.multiply %[[ARG0]], %[[T3]] : tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = chlo.erf %[[T4]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[T6:.*]] = mhlo.add %[[T5]], %[[T0]] : tensor<?x?xf32>
// CHECK:         %[[T7:.*]] = mhlo.multiply %[[T6]], %[[T2]] : tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = mhlo.multiply %[[ARG0]], %[[T7]] : tensor<?x?xf32>
// CHECK:         return %[[T8]] : tensor<?x?xf32>
func.func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %str = torch.constant.str "none"
    %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.sub.tensor(
// CHECK-SAME:            %[[ARG0:.*]]: tensor<?x?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:     %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:     %[[T1:.*]] = chlo.broadcast_multiply %[[ARG1]], %[[T0]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:     %[[T2:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T1]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:     return %[[T2]] : tensor<?x?x?x?xf32>
func.func @torch.aten.sub.tensor(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.sub.scalar.int(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?x?x4xf32>, %[[ARG1:.*]]: i64) -> tensor<?x?x?x4xf32> {
// CHECK:     %[[CST0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:     %[[T0:.*]] = tensor.from_elements %[[ARG1]] : tensor<1xi64>
// CHECK:     %[[T1:.*]] = mhlo.convert(%[[T0]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:     %[[T2:.*]] = "mhlo.reshape"(%[[T1]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:     %[[T3:.*]] = chlo.broadcast_multiply %3, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:     %[[T4:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T3]] : (tensor<?x?x?x4xf32>, tensor<f32>) -> tensor<?x?x?x4xf32>
// CHECK:     return %[[T4]] : tensor<?x?x?x4xf32>
func.func @torch.aten.sub.scalar.int(%arg0: !torch.vtensor<[?,?,?,4],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,?,?,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %int1 : !torch.vtensor<[?,?,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,4],f32>
  return %0 : !torch.vtensor<[?,?,?,4],f32>
}

// CHECK-LABEL:   func @torch.aten.add.scalar.float(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?x?x4xf32>, %[[ARG1:.*]]: f64) -> tensor<?x?x?x4xf32> {
// CHECK:     %[[CST0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:     %[[T0:.*]] = tensor.from_elements %[[ARG1]] : tensor<1xf64>
// CHECK:     %[[T1:.*]] = mhlo.convert(%[[T0]]) : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:     %[[T2:.*]] = "mhlo.reshape"(%[[T1]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:     %[[T3:.*]] = chlo.broadcast_multiply %[[T2]], %[[CST0]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:     %[[T4:.*]] = chlo.broadcast_add %[[ARG0]], %[[T3]] : (tensor<?x?x?x4xf32>, tensor<f32>) -> tensor<?x?x?x4xf32>
// CHECK:     return %[[T4]] : tensor<?x?x?x4xf32>
func.func @torch.aten.add.scalar.float(%arg0: !torch.vtensor<[?,?,?,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[?,?,?,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %arg1, %int1 : !torch.vtensor<[?,?,?,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,4],f32>
  return %0 : !torch.vtensor<[?,?,?,4],f32>
}

// CHECK-LABEL:   func @torch.aten.div.Tensor_mode(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:     %[[T0:.*]] = chlo.broadcast_divide %[[ARG0]], %[[ARG1]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:     %[[T1:.*]] = mhlo.sign %[[T0]] : tensor<?x?x?x?xf32>
// CHECK:     %[[T2:.*]] = mhlo.abs %[[T0]] : tensor<?x?x?x?xf32>
// CHECK:     %[[T3:.*]] = mhlo.floor %[[T2]] : tensor<?x?x?x?xf32>
// CHECK:     %[[T4:.*]] = mhlo.multiply %[[T1]], %[[T3]] : tensor<?x?x?x?xf32>
// CHECK:     return %[[T4]] : tensor<?x?x?x?xf32>
func.func @torch.aten.div.Tensor_mode(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %str = torch.constant.str "trunc"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.str -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.ne.scalar(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: f64) -> tensor<?x?xi1> {
// CHECK:     %[[T0:.*]] = tensor.from_elements %[[ARG1]] : tensor<1xf64>
// CHECK:     %[[T1:.*]] = mhlo.convert(%[[T0]]) : (tensor<1xf64>) -> tensor<1xi32>
// CHECK:     %[[T2:.*]] = "mhlo.reshape"(%[[T1]]) : (tensor<1xi32>) -> tensor<i32>
// CHECK:     %[[T3:.*]] = chlo.broadcast_compare %[[ARG0]], %[[T2]] {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction NE>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
// CHECK:     return %[[T3]] : tensor<?x?xi1>
func.func @torch.aten.ne.scalar(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.float) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.ne.Scalar %arg0, %arg1 : !torch.vtensor<[?,?],si32>, !torch.float -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// CHECK-LABEL:   func @torch.aten.ne.scalar.const(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?xi32>) -> tensor<?x?xi1> {
// CHECK:     %[[T0:.*]] = mhlo.constant dense<2> : tensor<i32>
// CHECK:     %[[T1:.*]] = chlo.broadcast_compare %[[ARG0]], %[[T0]] {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
// CHECK:     return %[[T1]] : tensor<?x?xi1>
func.func @torch.aten.ne.scalar.const(%arg0: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],i1> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.gt.Scalar %arg0, %float2.000000e00 : !torch.vtensor<[?,?],si32>, !torch.float -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// CHECK-LABEL:   func @torch.aten.gelu_backward(
// CHECK-SAME:            %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[CST_0:.*]] = "chlo.constant_like"(%[[ARG1]]) {value = 5.000000e-01 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CST_1:.*]] = "chlo.constant_like"(%[[ARG1]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CST_2:.*]] = "chlo.constant_like"(%[[ARG1]]) {value = 1.12837923 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CST_3:.*]] = "chlo.constant_like"(%[[ARG1]]) {value = 0.707106769 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[T0:.*]] = mhlo.multiply %[[CST_2]], %[[CST_3]] : tensor<?x?xf32>
// CHECK:           %[[T1:.*]] = mhlo.multiply %[[T0]], %[[CST_0]] : tensor<?x?xf32>
// CHECK:           %[[T2:.*]] = mhlo.multiply %[[ARG1]], %[[CST_3]] : tensor<?x?xf32>
// CHECK:           %[[T3:.*]] = chlo.erf %[[T2]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           %[[T4:.*]] = mhlo.negate %[[CST_0]] : tensor<?x?xf32>
// CHECK:           %[[T5:.*]] = mhlo.multiply %[[ARG1]], %[[T4]] : tensor<?x?xf32>
// CHECK:           %[[T6:.*]] = mhlo.add %[[CST_1]], %[[T3]] : tensor<?x?xf32>
// CHECK:           %[[T7:.*]] = mhlo.multiply %[[CST_0]], %[[T6]] : tensor<?x?xf32>
// CHECK:           %[[T8:.*]] = mhlo.multiply %[[ARG1]], %[[T5]] : tensor<?x?xf32>
// CHECK:           %[[T9:.*]] = mhlo.multiply %[[T8]], %[[T1]] : tensor<?x?xf32>
// CHECK:           %[[T10:.*]] = mhlo.add %[[T7]], %[[T9]] : tensor<?x?xf32>
// CHECK:           %[[T11:.*]] = mhlo.multiply %[[ARG1]], %[[T10]] : tensor<?x?xf32>
// CHECK:           return %[[T11]] : tensor<?x?xf32>
func.func @torch.aten.gelu_backward(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %str_none = torch.constant.str "none"
  %0 = torch.aten.gelu_backward %arg0, %arg1, %str_none: !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.native_dropout.train(
// CHECK-SAME:            %[[ARG0:.*]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi1>) {
// CHECK:           %[[CST_0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[CST_1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[CST_2:.*]] =  mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[SHAPE_0:.*]] = arith.constant 1 : index
// CHECK:           %[[SHAPE_1:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[SHAPE_1]] : tensor<?x?xf32>
// CHECK:           %[[DIM_0_I64:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:           %[[DIM_1:.*]] = tensor.dim %[[ARG0]], %[[SHAPE_0]] : tensor<?x?xf32>
// CHECK:           %[[DIM_1_I64:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:           %[[DIM_0_I32:.*]] = arith.trunci %[[DIM_0_I64]] : i64 to i32
// CHECK:           %[[DIM_1_I32:.*]] = arith.trunci %[[DIM_1_I64]] : i64 to i32
// CHECK:           %[[T0:.*]] = tensor.from_elements %[[DIM_0_I32]], %[[DIM_1_I32]] : tensor<2xi32>
// CHECK:           %[[T1:.*]] = "mhlo_disc.custom_call"(%[[CST_0]], %[[CST_2]], %[[T0]]) {backend_config = "{\22seed\22:1,\22seed2\22:2}", call_target_name = "rng_uniform", has_side_effect = false} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:           %[[T2:.*]] = mhlo.convert(%[[T1]]) : (tensor<?x?xf32>) -> tensor<?x?xf64>
// CHECK:           %[[T3:.*]] = chlo.broadcast_compare %[[T2]], %[[CST_1]] {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x?xf64>, tensor<f64>) -> tensor<?x?xi1>
// CHECK:           %[[T4:.*]] = mhlo.convert(%[[T3]]) : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[T5:.*]] = chlo.broadcast_multiply %[[T4]], %[[ARG0]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[T6:.*]] = chlo.broadcast_multiply %[[T5]], %[[CST_0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           %[[T7:.*]] = chlo.broadcast_compare %[[T4]], %[[CST_2]] {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:           return %[[T6]], %[[T7]] : tensor<?x?xf32>, tensor<?x?xi1>
func.func @torch.aten.native_dropout.train(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>) {
  %float1 = torch.constant.float 1.000000e+00
  %bool_true = torch.constant.bool true
  %result0, %result1 = torch.aten.native_dropout %arg0, %float1, %bool_true: !torch.vtensor<[?,?],f32>, !torch.float, !torch.bool -> !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
  return %result0, %result1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>
}