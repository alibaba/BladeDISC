// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.flip(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?xi64>) -> tensor<?x?x?xi64> {
// CHECK:         %[[T0:.*]] = "mhlo.reverse"(%[[ARG0]]) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
// CHECK:         return %[[T0]] : tensor<?x?x?xi64>
func.func @torch.aten.flip(%arg0: !torch.vtensor<[?,?,?],si64>) -> !torch.vtensor<[?,?,?],si64> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.flip %arg0, %0 : !torch.vtensor<[?,?,?],si64>, !torch.list<int> -> !torch.vtensor<[?,?,?],si64>
  return %1 : !torch.vtensor<[?,?,?],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index_select(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x4xf32>, %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x4xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 4]> : tensor<2xi32>
// CHECK:         %[[T0:.*]] = "mhlo.dynamic_gather"(%[[ARG0]], %[[ARG1]], %[[CST]]) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x4xf32>, tensor<2xi64>, tensor<2xi32>) -> tensor<2x4xf32>
// CHECK:         return %[[T0]] : tensor<2x4xf32>
func.func @torch.aten.index_select(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

