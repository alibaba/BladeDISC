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

// -----

// CHECK-LABEL:  func.func @torch.aten.roll(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<1> : tensor<2xi32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:         %[[C3_I64:.*]] = arith.constant 3 : i64
// CHECK:         %[[C:.*]]-9_i64 = arith.constant -9 : i64
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i64
// CHECK:         %[[T2:.*]] = arith.subi %[[T1]], %[[C3_I64]] : i64
// CHECK:         %[[T3:.*]] = arith.remsi %[[T2]], %[[T1]] : i64
// CHECK:         %[[T4:.*]] = arith.subi %[[C0_I64]], %[[T1]] : i64
// CHECK:         %[[T5:.*]] = arith.maxsi %[[T4]], %[[T3]] : i64
// CHECK:         %[[T6:.*]] = arith.minsi %[[T1]], %[[T5]] : i64
// CHECK:         %[[T7:.*]] = arith.addi %[[T1]], %[[T6]] : i64
// CHECK:         %[[T8:.*]] = arith.cmpi sge, %[[T6]], %[[C0_I64]] : i64
// CHECK:         %[[T9:.*]] = arith.select %[[T8]], %[[T6]], %[[T7]] : i64
// CHECK:         %[[T10:.*]] = arith.trunci %[[T9]] : i64 to i32
// CHECK:         %[[T11:.*]] = arith.trunci %[[T1]] : i64 to i32
// CHECK:         %[[T12:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T13:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T15:.*]] = arith.cmpi eq, %[[T11]], %[[C0_I32]] : i32
// CHECK:         %[[T16:.*]] = arith.select %[[T15]], %[[T14]], %[[T11]] : i32
// CHECK:         %[[T17:.*]] = tensor.from_elements %[[C0_I32]], %[[T10]] : tensor<2xi32>
// CHECK:         %[[T18:.*]] = tensor.from_elements %[[T13]], %[[T16]] : tensor<2xi32>
// CHECK:         %[[T19:.*]] = mhlo.real_dynamic_slice %[[ARG0]], %[[T17]], %[[T18]], %[[CST]] : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T20:.*]] = arith.maxsi %[[T4]], %[[C0_I64]] : i64
// CHECK:         %[[T21:.*]] = arith.minsi %[[T1]], %[[T20]] : i64
// CHECK:         %[[T22:.*]] = arith.addi %[[T1]], %[[T21]] : i64
// CHECK:         %[[T23:.*]] = arith.cmpi sge, %[[T21]], %[[C0_I64]] : i64
// CHECK:         %[[T24:.*]] = arith.select %[[T23]], %[[T21]], %[[T22]] : i64
// CHECK:         %[[T25:.*]] = arith.trunci %[[T24]] : i64 to i32
// CHECK:         %[[T26:.*]] = arith.cmpi eq, %[[T10]], %[[C0_I32]] : i32
// CHECK:         %[[T27:.*]] = arith.select %[[T26]], %[[T14]], %[[T10]] : i32
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[C0_I32]], %[[T25]] : tensor<2xi32>
// CHECK:         %[[T29:.*]] = tensor.from_elements %[[T13]], %[[T27]] : tensor<2xi32>
// CHECK:         %[[T30:.*]] = mhlo.real_dynamic_slice %[[ARG0]], %[[T28]], %[[T29]], %[[CST]] : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T31:.*]] = "mhlo.concatenate"(%[[T19]], %[[T30]]) {dimension = 1 : i64} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T32:.*]] = tensor.dim %[[T31]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T33:.*]] = arith.index_cast %[[T32]] : index to i64
// CHECK:         %[[T34:.*]] = arith.subi %[[T33]], %[[C]]-9_i64 : i64
// CHECK:         %[[T35:.*]] = arith.remsi %[[T34]], %[[T33]] : i64
// CHECK:         %[[T36:.*]] = arith.subi %[[C0_I64]], %[[T33]] : i64
// CHECK:         %[[T37:.*]] = arith.maxsi %[[T36]], %[[T35]] : i64
// CHECK:         %[[T38:.*]] = arith.minsi %[[T33]], %[[T37]] : i64
// CHECK:         %[[T39:.*]] = arith.addi %[[T33]], %[[T38]] : i64
// CHECK:         %[[T40:.*]] = arith.cmpi sge, %[[T38]], %[[C0_I64]] : i64
// CHECK:         %[[T41:.*]] = arith.select %[[T40]], %[[T38]], %[[T39]] : i64
// CHECK:         %[[T42:.*]] = arith.trunci %[[T41]] : i64 to i32
// CHECK:         %[[T43:.*]] = arith.trunci %[[T33]] : i64 to i32
// CHECK:         %[[T44:.*]] = arith.index_cast %[[T32]] : index to i32
// CHECK:         %[[T45:.*]] = tensor.dim %[[T31]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T46:.*]] = arith.index_cast %[[T45]] : index to i32
// CHECK:         %[[T47:.*]] = arith.cmpi eq, %[[T43]], %[[C0_I32]] : i32
// CHECK:         %[[T48:.*]] = arith.select %[[T47]], %[[T44]], %[[T43]] : i32
// CHECK:         %[[T49:.*]] = tensor.from_elements %[[T42]], %[[C0_I32]] : tensor<2xi32>
// CHECK:         %[[T50:.*]] = tensor.from_elements %[[T48]], %[[T46]] : tensor<2xi32>
// CHECK:         %[[T51:.*]] = mhlo.real_dynamic_slice %[[T31]], %[[T49]], %[[T50]], %[[CST]] : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T52:.*]] = arith.maxsi %[[T36]], %[[C0_I64]] : i64
// CHECK:         %[[T53:.*]] = arith.minsi %[[T33]], %[[T52]] : i64
// CHECK:         %[[T54:.*]] = arith.addi %[[T33]], %[[T53]] : i64
// CHECK:         %[[T55:.*]] = arith.cmpi sge, %[[T53]], %[[C0_I64]] : i64
// CHECK:         %[[T56:.*]] = arith.select %[[T55]], %[[T53]], %[[T54]] : i64
// CHECK:         %[[T57:.*]] = arith.trunci %[[T56]] : i64 to i32
// CHECK:         %[[T58:.*]] = arith.cmpi eq, %[[T42]], %[[C0_I32]] : i32
// CHECK:         %[[T59:.*]] = arith.select %[[T58]], %[[T44]], %[[T42]] : i32
// CHECK:         %[[T60:.*]] = tensor.from_elements %[[T57]], %[[C0_I32]] : tensor<2xi32>
// CHECK:         %[[T61:.*]] = tensor.from_elements %[[T59]], %[[T46]] : tensor<2xi32>
// CHECK:         %[[T62:.*]] = mhlo.real_dynamic_slice %[[T31]], %[[T60]], %[[T61]], %[[CST]] : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:         %[[T63:.*]] = "mhlo.concatenate"(%[[T51]], %[[T62]]) {dimension = 0 : i64} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         return %[[T63]] : tensor<?x?xf32>
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

