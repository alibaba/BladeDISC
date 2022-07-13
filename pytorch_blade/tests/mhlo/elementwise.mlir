// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.sigmoid(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = mhlo.negate %[[ARG0]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = mhlo.exponential %[[T1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.add %[[T2]], %[[T0]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.divide %[[T0]], %[[T3]] : tensor<?x?xf32>
// CHECK:         return %[[T4]] : tensor<?x?xf32>
func @torch.aten.sigmoid(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func @torch.aten.relu(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = "chlo.constant_like"(%[[ARG0]]) {value = 0.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = "mhlo.compare"(%[[ARG0]], %[[T0]]) {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:         %[[T2:.*]] = "mhlo.select"(%[[T1]], %[[ARG0]], %[[T0]]) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T2]] : tensor<?x?xf32>
func @torch.aten.relu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
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
func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
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
func @torch.aten.sub.tensor(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

func @torch.aten.sub.tensor(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],si32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],si32>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
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
func @torch.aten.sub.scalar.int(%arg0: !torch.vtensor<[?,?,?,4],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,?,?,4],f32> {
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
func @torch.aten.add.scalar.float(%arg0: !torch.vtensor<[?,?,?,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[?,?,?,4],f32> {
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
func @torch.aten.div.Tensor_mode(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %str = torch.constant.str "trunc"
  %0 = torch.aten.div.Tensor_mode %arg0, %arg1, %str : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.str -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.ne.scalar(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: f64) -> tensor<?x?xi1> {
// CHECK:     %[[T0:.*]] = tensor.from_elements %[[ARG1]] : tensor<1xf64>
// CHECK:     %[[T1:.*]] = mhlo.convert(%[[T0]]) : (tensor<1xf64>) -> tensor<1xi32>
// CHECK:     %[[T2:.*]] = "mhlo.reshape"(%[[T1]]) : (tensor<1xi32>) -> tensor<i32>
// CHECK:     %[[T3:.*]] = chlo.broadcast_compare %[[ARG0]], %[[T2]] {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
// CHECK:     return %[[T3]] : tensor<?x?xi1>
func @torch.aten.ne.scalar(%arg0: !torch.vtensor<[?,?],si32>, %arg1: !torch.float) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.ne.Scalar %arg0, %arg1 : !torch.vtensor<[?,?],si32>, !torch.float -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// CHECK-LABEL:   func @torch.aten.ne.scalar.const(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?xi32>) -> tensor<?x?xi1> {
// CHECK:     %[[T0:.*]] = mhlo.constant dense<2> : tensor<i32>
// CHECK:     %[[T1:.*]] = chlo.broadcast_compare %[[ARG0]], %[[T0]] {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
// CHECK:     return %[[T1]] : tensor<?x?xi1>
func @torch.aten.ne.scalar.const(%arg0: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],i1> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.gt.Scalar %arg0, %float2.000000e00 : !torch.vtensor<[?,?],si32>, !torch.float -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}
