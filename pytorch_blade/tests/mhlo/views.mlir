// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.aten.view(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x224xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[-1, 224]> : tensor<2xi32>
// CHECK:         %[[T0:.*]] = "chlo.dynamic_reshape"(%[[ARG0]], %[[CST]]) : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x224xf32>
// CHECK:         return %[[T0]] : tensor<?x224xf32>
func @torch.aten.view(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
  %int-1 = torch.constant.int -1
  %int224 = torch.constant.int 224
  %0 = torch.prim.ListConstruct %int-1, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,224],f32>
  return %1 : !torch.vtensor<[?,224],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.reshape(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?x?xf32>) -> tensor<?x120x4x64xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[-1, 120, 4, 64]> : tensor<4xi32>
// CHECK:         %[[T0:.*]] = "chlo.dynamic_reshape"(%[[ARG0]], %[[CST]]) : (tensor<?x?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x120x4x64xf32>
// CHECK:         return %[[T0]] : tensor<?x120x4x64xf32>
func @torch.aten.reshape(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
  %int-1 = torch.constant.int -1
  %int120 = torch.constant.int 120
  %int4 = torch.constant.int 4
  %int64 = torch.constant.int 64
  %0 = torch.prim.ListConstruct %int-1, %int120, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reshape %arg0, %0 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,120,4,64],f32>
  return %1 : !torch.vtensor<[?,120,4,64],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.flatten(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
// CHECK:         %[[T0:.*]] = "chlo.dynamic_reshape"(%[[ARG0]], %[[CST]]) : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK:         return %[[T0]] : tensor<?xf32>
func @torch.aten.flatten(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %1 = torch.aten.flatten.using_ints %arg0, %int0, %int3: !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
  return %1 : !torch.vtensor<[?],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.view.minus1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x?x?xf32>) -> tensor<2x3x?xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[2, 3, -1]> : tensor<3xi32>
// CHECK:         %[[T0:.*]] = "chlo.dynamic_reshape"(%[[ARG0]], %[[CST]]) : (tensor<2x3x?x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK:         return %[[T0]] : tensor<2x3x?xf32>
func @torch.aten.view.minus1(%arg0: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
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
// CHECK:         module {
// CHECK-LABEL:  func @torch.aten.view.to_rank1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<f32>) -> tensor<1xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:         %[[T0:.*]] = "chlo.dynamic_reshape"(%[[ARG0]], %[[CST]]) : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK:         return %[[T0]] : tensor<1xf32>
func @torch.aten.view.to_rank1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}

// -----
// CHECK-LABEL:  func @torch.aten.view.to_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1xf32>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         return %[[T0]] : tensor<f32>
func @torch.aten.view.to_rank0(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}

