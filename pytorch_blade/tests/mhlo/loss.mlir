// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.nll_loss_forward(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?xi32>) -> (tensor<f32>, tensor<f32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[T3:.*]] = arith.sitofp %[[T2]] : i64 to f64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]] : tensor<1xf64>
// CHECK:         %[[T5:.*]] = mhlo.convert %[[T4]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK:         %[[T6:.*]] = mhlo.reshape %[[T5]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T1]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T7]], %[[C1_I32]] : tensor<2xi32>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_gather"(%[[ARG0]], %[[ARG1]], %[[T8]]) {dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x?xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = mhlo.reduce(%[[T9]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_divide %[[T10]], %[[T6]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T11]], %[[T6]] : tensor<f32>, tensor<f32>
func.func @torch.aten.nll_loss_forward(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],si32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>){
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int_100 = torch.constant.int -100
  %result0, %result1 = torch.aten.nll_loss_forward %arg0, %arg1, %none, %int1, %int_100 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],si32>, !torch.none, !torch.int, !torch.int -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  return %result0, %result1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
}

