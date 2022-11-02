// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.view(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x224xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[-1, 224]> : tensor<2xi32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = arith.muli %[[T1]], %[[T3]] : i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = arith.muli %[[T4]], %[[T6]] : i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = arith.muli %[[T7]], %[[T9]] : i32
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : i32 to index
// CHECK:         %[[T12:.*]] = mhlo.compute_reshape_shape %[[T11]], %[[CST]] : (index, tensor<2xi32>) -> tensor<2xi32>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T12]] : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x224xf32>
// CHECK:         return %[[T13]] : tensor<?x224xf32>
func.func @torch.aten.view(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
  %int-1 = torch.constant.int -1
  %int224 = torch.constant.int 224
  %0 = torch.prim.ListConstruct %int-1, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,224],f32>
  return %1 : !torch.vtensor<[?,224],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.reshape(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?x?xf32>) -> tensor<?x120x4x64xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[-1, 120, 4, 64]> : tensor<4xi32>
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = arith.muli %[[T1]], %[[T3]] : i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = arith.muli %[[T4]], %[[T6]] : i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = arith.muli %[[T7]], %[[T9]] : i32
// CHECK:         %[[T11:.*]] = tensor.dim %[[ARG0]], %[[C4]] : tensor<?x?x?x?x?xf32>
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T11]] : index to i32
// CHECK:         %[[T13:.*]] = arith.muli %[[T10]], %[[T12]] : i32
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T13]] : i32 to index
// CHECK:         %[[T15:.*]] = mhlo.compute_reshape_shape %[[T14]], %[[CST]] : (index, tensor<4xi32>) -> tensor<4xi32>
// CHECK:         %[[T16:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T15]] : (tensor<?x?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x120x4x64xf32>
// CHECK:         return %[[T16]] : tensor<?x120x4x64xf32>
func.func @torch.aten.reshape(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
  %int-1 = torch.constant.int -1
  %int120 = torch.constant.int 120
  %int4 = torch.constant.int 4
  %int64 = torch.constant.int 64
  %0 = torch.prim.ListConstruct %int-1, %int120, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reshape %arg0, %0 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,120,4,64],f32>
  return %1 : !torch.vtensor<[?,120,4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.flatten(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = arith.muli %[[T1]], %[[T3]] : i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = arith.muli %[[T4]], %[[T6]] : i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = arith.muli %[[T7]], %[[T9]] : i32
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : i32 to index
// CHECK:         %[[T12:.*]] = mhlo.compute_reshape_shape %[[T11]], %[[CST]] : (index, tensor<1xi32>) -> tensor<1xi32>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T12]] : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK:         return %[[T13]] : tensor<?xf32>
func.func @torch.aten.flatten(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %1 = torch.aten.flatten.using_ints %arg0, %int0, %int3: !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  return %1 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view.minus1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x?x?xf32>) -> tensor<2x3x?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[2, 3, -1]> : tensor<3xi32>
// CHECK:         %[[C6_I32:.*]] = arith.constant 6 : i32
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<2x3x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = arith.muli %[[T1]], %[[C6_I32]] : i32
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<2x3x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = arith.muli %[[T2]], %[[T4]] : i32
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : i32 to index
// CHECK:         %[[T7:.*]] = mhlo.compute_reshape_shape %[[T6]], %[[CST]] : (index, tensor<3xi32>) -> tensor<3xi32>
// CHECK:         %[[T8:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T7]] : (tensor<2x3x?x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK:         return %[[T8]] : tensor<2x3x?xf32>
func.func @torch.aten.view.minus1(%arg0: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
  %int-1 = torch.constant.int -1
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %2 = torch.prim.ListConstruct %0, %1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[2,3,?,?],f32>, !torch.list<int> -> !torch.vtensor<[2,3,?],f32>
  return %3 : !torch.vtensor<[2,3,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view.to_rank1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<f32>) -> tensor<1xf32> {
// CHECK:         %[[T0:.*]] = mhlo.reshape %[[ARG0]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:         return %[[T0]] : tensor<1xf32>
func.func @torch.aten.view.to_rank1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view.to_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.reshape %[[ARG0]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         return %[[T0]] : tensor<f32>
func.func @torch.aten.view.to_rank0(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}

