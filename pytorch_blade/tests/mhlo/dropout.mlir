// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

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

// CHECK-LABEL:  func @torch.aten.native_dropout_backward(
// CHECK-SAME:              %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xi1>) -> tensor<?x?xf32> {
// CHECK:           %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[T1:.*]] = mhlo.convert(%[[ARG1]]) : (tensor<?x?xi1>) -> tensor<?x?xf32>
// CHECK:           %[[T2:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[T1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[T3:.*]] = chlo.broadcast_multiply %[[T2]], %[[T0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:           return %[[T3]] : tensor<?x?xf32>
// CHECK:         }
func.func @torch.aten.native_dropout_backward(%arg0: !torch.vtensor<[?,?],f32>, %arg1 : !torch.vtensor<[?,?],i1>) -> !torch.vtensor<[?,?],f32>{
  %float_1 = torch.constant.float 1.000000
  %result = torch.aten.native_dropout_backward %arg0, %arg1, %float_1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],i1>, !torch.float -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}