// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.nll_loss_forward(
// CHECK-SAME:              %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?xi32>) -> (tensor<f32>, tensor<f32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[CST_1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[ARG0]], %[[CST_0]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1:.*]] : index to i32
// CHECK:         %[[T3:.*]] = tensor.from_elements %[[T2]] : tensor<1xi32>
// CHECK:         %[[T4:.*]] = mhlo.convert(%[[T3]]) : (tensor<1xi32>) -> tensor<1xf32>
// CHECK:         %[[T5:.*]] = mhlo.reshape %[[T4]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T2]], %[[CST_1_I32]] : tensor<2xi32>
// CHECK:         %[[T7:.*]] = "mhlo.dynamic_gather"(%[[ARG0]], %[[ARG1]], %[[T6]]) {dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x?xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = mhlo.reduce(%[[T7]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T9:.*]] = mhlo.divide %[[T8]], %[[T5]] : tensor<f32>
// CHECK:         return %[[T9]], %[[T5]] : tensor<f32>, tensor<f32>
// CHECK:       }
func.func @torch.aten.nll_loss_forward(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],si32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>){
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int_100 = torch.constant.int -100
  %result0, %result1 = torch.aten.nll_loss_forward %arg0, %arg1, %none, %int1, %int_100 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],si32>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  return %result0, %result1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
}