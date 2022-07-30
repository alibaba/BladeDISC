// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.flip(
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

// CHECK-LABEL:  func @torch.aten.index_select(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x4xf32>, %[[ARG1:.*]]: tensor<2xi64>) -> tensor<2x4xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1, 4]> : tensor<2xi32>
// CHECK:         %[[T0:.*]] = "mhlo.dynamic_gather"(%[[ARG0]], %[[ARG1]], %[[CST]]) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x4xf32>, tensor<2xi64>, tensor<2xi32>) -> tensor<2x4xf32>
// CHECK:         return %[[T0]] : tensor<2x4xf32>
func.func @torch.aten.index_select(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.index_select %arg0, %int0, %arg1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL:  func @torch.aten.roll(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[C:.*]]-9_i32 = arith.constant -9 : i32
// CHECK:         %[[CST:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<1> : tensor<2xi32>
// CHECK:         %[[C3_I32:.*]] = arith.constant 3 : i32
// CHECK:         %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = arith.subi %[[T3]], %[[C3_I32]] : i32
// CHECK:         %[[T5:.*]] = arith.remsi %[[T4]], %[[T3]] : i32
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[C0_I32]], %[[T5]] : tensor<2xi32>
// CHECK:         %[[T7:.*]] = tensor.from_elements %[[T1]], %[[T3]] : tensor<2xi32>
// CHECK:         %[[T8:.*]] = "mhlo.real_dynamic_slice"(%[[ARG0]], %[[T6]], %[[T7]], %[[CST_0]]) : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = arith.cmpi eq, %[[T5]], %[[C0_I32]] : i32
// CHECK:         %[[T10:.*]] = arith.select %[[T9]], %[[T3]], %[[T5]] : i32
// CHECK:         %[[T11:.*]] = tensor.from_elements %[[T1]], %[[T1]]0 : tensor<2xi32>
// CHECK:         %[[T12:.*]] = "mhlo.real_dynamic_slice"(%[[ARG0]], %[[CST]], %[[T11]], %[[CST]]_0) : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T13:.*]] = "mhlo.concatenate"(%[[T8]], %[[T12]]) {dimension = 1 : i64} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T14:.*]] = tensor.dim %[[T13]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T15:.*]] = arith.index_cast %[[T14]] : index to i32
// CHECK:         %[[T16:.*]] = tensor.dim %[[T13]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[T16]] : index to i32
// CHECK:         %[[T18:.*]] = arith.subi %[[T15]], %[[C]]-9_i32 : i32
// CHECK:         %[[T19:.*]] = arith.remsi %[[T18]], %[[T15]] : i32
// CHECK:         %[[T20:.*]] = tensor.from_elements %[[T19]], %[[C0_I32]] : tensor<2xi32>
// CHECK:         %[[T21:.*]] = tensor.from_elements %[[T15]], %[[T17]] : tensor<2xi32>
// CHECK:         %[[T22:.*]] = "mhlo.real_dynamic_slice"(%[[T13]], %[[T20]], %[[T21]], %[[CST_0]]) : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T23:.*]] = arith.cmpi eq, %[[T19]], %[[C0_I32]] : i32
// CHECK:         %[[T24:.*]] = arith.select %[[T23]], %[[T15]], %[[T19]] : i32
// CHECK:         %[[T25:.*]] = tensor.from_elements %[[T24]], %[[T17]] : tensor<2xi32>
// CHECK:         %[[T26:.*]] = "mhlo.real_dynamic_slice"(%[[T13]], %[[CST]], %[[T25]], %[[CST]]_0) : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T27:.*]] = "mhlo.concatenate"(%[[T22]], %[[T26]]) {dimension = 0 : i64} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T27]] : tensor<?x?xf32>
func.func @torch.aten.roll(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-9 = torch.constant.int -9
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int-9 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.roll %arg0, %0, %1 : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %2 : !torch.vtensor<[?,?],f32>
}
