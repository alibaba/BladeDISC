// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.sum.div.Scalar(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = arith.muli %[[T3]], %[[T5]] : i32
// CHECK:         %[[T11:.*]] = arith.muli %[[T10]], %[[T7]] : i32
// CHECK:         %[[T12:.*]] = arith.muli %[[T11]], %[[T9]] : i32
// CHECK:         %[[T13:.*]] = arith.extsi %[[T12]] : i32 to i64
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T13]] : tensor<1xi64>
// CHECK:         %[[T15:.*]] = mhlo.convert(%[[T14]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T16:.*]] = "mhlo.reshape"(%[[T15]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_divide %[[T1]], %[[T1]]6 : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T17]] : tensor<f32>
func.func @torch.aten.sum.div.Scalar(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f32>
  %1 = torch.aten.numel %arg0 : !torch.vtensor<[?,?,?,?],f32> -> !torch.int
  %2 = torch.aten.div.Scalar %0, %1 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %2 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:  func @torch.aten.sum.div.Scalar.si32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xi32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xi32>, tensor<i32>) -> tensor<i32>
// CHECK:         %[[T2:.*]] = mhlo.convert(%[[T1]]) : (tensor<i32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i32
// CHECK:         %[[T9:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : index to i32
// CHECK:         %[[T11:.*]] = arith.muli %[[T4]], %[[T6]] : i32
// CHECK:         %[[T12:.*]] = arith.muli %[[T11]], %[[T8]] : i32
// CHECK:         %[[T13:.*]] = arith.muli %[[T12]], %[[T10]] : i32
// CHECK:         %[[T14:.*]] = arith.extsi %[[T13]] : i32 to i64
// CHECK:         %[[T15:.*]] = tensor.from_elements %[[T14]] : tensor<1xi64>
// CHECK:         %[[T16:.*]] = mhlo.convert(%[[T15]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T17:.*]] = "mhlo.reshape"(%[[T16]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_divide %[[T2]], %[[T17]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T18]] : tensor<f32>
func.func @torch.aten.sum.div.Scalar.si32(%arg0: !torch.vtensor<[?,?,?,?],si32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],si32>, !torch.int -> !torch.vtensor<[],f32>
  %1 = torch.aten.numel %arg0 : !torch.vtensor<[?,?,?,?],si32> -> !torch.int
  %2 = torch.aten.div.Scalar %0, %1 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %2 : !torch.vtensor<[],f32>
}


// CHECK-LABEL:  func @torch.aten.sum.outf32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T1]] : tensor<f32>
func.func @torch.aten.sum.outf32(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}


// CHECK-LABEL:  func @torch.aten.sum.outf64(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f64> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.convert(%[[T1]]) : (tensor<f32>) -> tensor<f64>
// CHECK:         return %[[T2]] : tensor<f64>
func.func @torch.aten.sum.outf64(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f64> {
  %int7 = torch.constant.int 7
  %0 = torch.aten.sum %arg0, %int7 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}


// CHECK-LABEL:  func @torch.aten.sum(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T1]] : tensor<f32>
func.func @torch.aten.sum(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}


// CHECK-LABEL:  func @torch.aten.sum.f64(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf64>) -> tensor<f64> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf64>, tensor<f64>) -> tensor<f64>
// CHECK:         return %[[T1]] : tensor<f64>
func.func @torch.aten.sum.f64(%arg0: !torch.vtensor<[?,?,?,?],f64>) -> !torch.vtensor<[],f64> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f64>, !torch.none -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}


// CHECK-LABEL:  func @torch.aten.sum.si32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xi32>) -> tensor<i32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xi32>, tensor<i32>) -> tensor<i32>
// CHECK:         return %[[T1]] : tensor<i32>
func.func @torch.aten.sum.si32(%arg0: !torch.vtensor<[?,?,?,?],si32>) -> !torch.vtensor<[],si32> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.sum %arg0, %int3 : !torch.vtensor<[?,?,?,?],si32>, !torch.int -> !torch.vtensor<[],si32>
  return %0 : !torch.vtensor<[],si32>
}


// CHECK-LABEL:  func @torch.aten.sum.dim_IntList(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x?x?x?xf32>) -> tensor<2xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [1, 2, 3] : (tensor<2x?x?x?xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK:         return %[[T1]] : tensor<2xf32>
func.func @torch.aten.sum.dim_IntList(%arg0: !torch.vtensor<[2,?,?,?],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[2,?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2],f32>
  return %1 : !torch.vtensor<[2],f32>
}


// CHECK-LABEL:  func @torch.aten.sum.dim_IntList.keepdim(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x?x224x?xf32>) -> tensor<2x1x224x1xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [1, 3] : (tensor<2x?x224x?xf32>, tensor<f32>) -> tensor<2x224xf32>
// CHECK:         %[[T2:.*]] = "mhlo.reshape"(%[[T1]]) : (tensor<2x224xf32>) -> tensor<2x1x224x1xf32>
// CHECK:         return %[[T2]] : tensor<2x1x224x1xf32>
func.func @torch.aten.sum.dim_IntList.keepdim(%arg0: !torch.vtensor<[2,?,224,?],f32>) -> !torch.vtensor<[2,1,224,1],f32> {
  %none = torch.constant.none
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[2,?,224,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,1,224,1],f32>
  return %1 : !torch.vtensor<[2,1,224,1],f32>
}

