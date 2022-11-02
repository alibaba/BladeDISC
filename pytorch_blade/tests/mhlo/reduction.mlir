// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.sum.div.Scalar(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = arith.muli %[[T3]], %[[T5]] : i32
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i32
// CHECK:         %[[T9:.*]] = arith.muli %[[T6]], %[[T8]] : i32
// CHECK:         %[[T10:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T12:.*]] = arith.muli %[[T9]], %[[T11]] : i32
// CHECK:         %[[T13:.*]] = arith.extsi %[[T12]] : i32 to i64
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T13]] : tensor<1xi64>
// CHECK:         %[[T15:.*]] = mhlo.convert %[[T14]] : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T16:.*]] = mhlo.reshape %[[T15]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_divide %[[T1]], %[[T16]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T17]] : tensor<f32>
func.func @torch.aten.sum.div.Scalar(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f32>
  %1 = torch.aten.numel %arg0 : !torch.vtensor<[?,?,?,?],f32> -> !torch.int
  %2 = torch.aten.div.Scalar %0, %1 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %2 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.div.Scalar.si32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xi32>) -> tensor<f32> {
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.convert %[[ARG0]] : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[T1]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = arith.muli %[[T4]], %[[T6]] : i32
// CHECK:         %[[T8:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i32
// CHECK:         %[[T10:.*]] = arith.muli %[[T7]], %[[T9]] : i32
// CHECK:         %[[T11:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xi32>
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T11]] : index to i32
// CHECK:         %[[T13:.*]] = arith.muli %[[T10]], %[[T12]] : i32
// CHECK:         %[[T14:.*]] = arith.extsi %[[T13]] : i32 to i64
// CHECK:         %[[T15:.*]] = tensor.from_elements %[[T14]] : tensor<1xi64>
// CHECK:         %[[T16:.*]] = mhlo.convert %[[T15]] : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T17:.*]] = mhlo.reshape %[[T16]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_divide %[[T2]], %[[T17]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T18]] : tensor<f32>
func.func @torch.aten.sum.div.Scalar.si32(%arg0: !torch.vtensor<[?,?,?,?],si32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],si32>, !torch.int -> !torch.vtensor<[],f32>
  %1 = torch.aten.numel %arg0 : !torch.vtensor<[?,?,?,?],si32> -> !torch.int
  %2 = torch.aten.div.Scalar %0, %1 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %2 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.outf32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T1]] : tensor<f32>
func.func @torch.aten.sum.outf32(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.aten.sum %arg0, %int6 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.outf64(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f64> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[T1:.*]] = mhlo.convert %[[ARG0]] : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf64>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[T1]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf64>, tensor<f64>) -> tensor<f64>
// CHECK:         return %[[T2]] : tensor<f64>
func.func @torch.aten.sum.outf64(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f64> {
  %int7 = torch.constant.int 7
  %0 = torch.aten.sum %arg0, %int7 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         return %[[T1]] : tensor<f32>
func.func @torch.aten.sum(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.f64(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf64>) -> tensor<f64> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xf64>, tensor<f64>) -> tensor<f64>
// CHECK:         return %[[T1]] : tensor<f64>
func.func @torch.aten.sum.f64(%arg0: !torch.vtensor<[?,?,?,?],f64>) -> !torch.vtensor<[],f64> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?,?],f64>, !torch.none -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.si32(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xi32>) -> tensor<i32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1, 2, 3] : (tensor<?x?x?x?xi32>, tensor<i32>) -> tensor<i32>
// CHECK:         return %[[T1]] : tensor<i32>
func.func @torch.aten.sum.si32(%arg0: !torch.vtensor<[?,?,?,?],si32>) -> !torch.vtensor<[],si32> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.sum %arg0, %int3 : !torch.vtensor<[?,?,?,?],si32>, !torch.int -> !torch.vtensor<[],si32>
  return %0 : !torch.vtensor<[],si32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.dim_IntList(
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

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.dim_IntList.keepdim(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x?x224x?xf32>) -> tensor<2x1x224x1xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [1, 3] : (tensor<2x?x224x?xf32>, tensor<f32>) -> tensor<2x224xf32>
// CHECK:         %[[T2:.*]] = mhlo.reshape %[[T1]] : (tensor<2x224xf32>) -> tensor<2x1x224x1xf32>
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

// -----

// CHECK-LABEL:  func.func @torch.aten.max.dim(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> (tensor<2x3x224x1xf32>, tensor<2x3x224x1xi32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T1]]) applies mhlo.maximum across dimensions = [3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3x224xf32>
// CHECK:         %[[T3:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<224xi32>
// CHECK:         %[[T4:.*]] = "mhlo.broadcast_in_dim"(%[[T3]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<224xi32>) -> tensor<2x3x224x224xi32>
// CHECK:         %[[T5:.*]]:2 = mhlo.reduce(%[[ARG0]] init: %[[T1]]), (%[[T4]] init: %[[T0]]) across dimensions = [3] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x3x224xf32>, tensor<2x3x224xi32>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i32>, %[[ARG4:.*]]: tensor<i32>)  {
// CHECK:         %[[T8:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T9:.*]] = mhlo.select %[[T8]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T10:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T11:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i32>
// CHECK:         %[[T12:.*]] = mhlo.select %[[T8]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i32>
// CHECK:         %[[T13:.*]] = mhlo.select %[[T10]], %[[T11]], %[[T12]] : tensor<i1>, tensor<i32>
// CHECK:         mhlo.return %[[T9]], %[[T13]] : tensor<f32>, tensor<i32>
// CHECK:         %[[T6:.*]] = mhlo.reshape %[[T2]] : (tensor<2x3x224xf32>) -> tensor<2x3x224x1xf32>
// CHECK:         %[[T7:.*]] = mhlo.reshape %[[T5]]#1 : (tensor<2x3x224xi32>) -> tensor<2x3x224x1xi32>
// CHECK:         return %[[T6]], %[[T7]] : tensor<2x3x224x1xf32>, tensor<2x3x224x1xi32>
func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[2,3,224,224],f32>) -> (!torch.vtensor<[2,3,224,1],f32>, !torch.vtensor<[2,3,224,1],si32>){
  %none = torch.constant.none
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.max.dim %arg0, %int-1, %true : !torch.vtensor<[2,3,224,224],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,3,224,1],f32>, !torch.vtensor<[2,3,224,1],si32>
  return %values, %indices : !torch.vtensor<[2,3,224,1],f32>, !torch.vtensor<[2,3,224,1],si32>
}

