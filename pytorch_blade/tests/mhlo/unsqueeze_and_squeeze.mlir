// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim0(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x1x2x1x2xf32>) -> tensor<2x1x2x1x2xf32> {
// CHECK:         return %[[ARG0]] : tensor<2x1x2x1x2xf32>
func.func @torch.aten.squeeze.dim0(%arg0: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,1,2,1,2],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.squeeze.dim %arg0, %int0 : !torch.vtensor<[2,1,2,1,2],f32>, !torch.int -> !torch.vtensor<[2,1,2,1,2],f32>
  return %0 : !torch.vtensor<[2,1,2,1,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x1x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T1]], %[[T3]], %[[C1_I32]], %[[T5]] : tensor<4xi32>
// CHECK:         %[[T7:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T6]] : (tensor<?x1x?x1x?xf32>, tensor<4xi32>) -> tensor<?x?x1x?xf32>
// CHECK:         return %[[T7]] : tensor<?x?x1x?xf32>
func.func @torch.aten.squeeze.dim1(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,?,1,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.squeeze.dim %arg0, %int1 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim2.from_end(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x1x?x1x?xf32>) -> tensor<?x1x?x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T1]], %[[C1_I32]], %[[T3]], %[[T5]] : tensor<4xi32>
// CHECK:         %[[T7:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T6]] : (tensor<?x1x?x1x?xf32>, tensor<4xi32>) -> tensor<?x1x?x?xf32>
// CHECK:         return %[[T7]] : tensor<?x1x?x?xf32>
func.func @torch.aten.squeeze.dim2.from_end(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,1,?,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.squeeze.dim %arg0, %int-2 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x1x2x1x2xf32>) -> tensor<2x2x2xf32> {
// CHECK:         %[[T0:.*]] = mhlo.reshape %[[ARG0]] : (tensor<2x1x2x1x2xf32>) -> tensor<2x2x2xf32>
// CHECK:         return %[[T0]] : tensor<2x2x2xf32>
func.func @torch.aten.squeeze(%arg0: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[2,1,2,1,2],f32> -> !torch.vtensor<[2,2,2],f32>
  return %0 : !torch.vtensor<[2,2,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze.dim0(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<1x?x?x?x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[C1_I32]], %[[T1]], %[[T3]], %[[T5]], %[[T7]] : tensor<5xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T8]] : (tensor<?x?x?x?xf32>, tensor<5xi32>) -> tensor<1x?x?x?x?xf32>
// CHECK:         return %[[T9]] : tensor<1x?x?x?x?xf32>
func.func @torch.aten.unsqueeze.dim0(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1,?,?,?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[1,?,?,?,?],f32>
  return %0 : !torch.vtensor<[1,?,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze.dim1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x1x?x?x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T1]], %[[C1_I32]], %[[T3]], %[[T5]], %[[T7]] : tensor<5xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T8]] : (tensor<?x?x?x?xf32>, tensor<5xi32>) -> tensor<?x1x?x?x?xf32>
// CHECK:         return %[[T9]] : tensor<?x1x?x?x?xf32>
func.func @torch.aten.unsqueeze.dim1(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,1,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.unsqueeze %arg0, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze.dim2.from_end(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x1x?xf32> {
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[T0]] : index to i32
// CHECK:         %[[T2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i32
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T1]], %[[T3]], %[[T5]], %[[C1_I32]], %[[T7]] : tensor<5xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T8]] : (tensor<?x?x?x?xf32>, tensor<5xi32>) -> tensor<?x?x?x1x?xf32>
// CHECK:         return %[[T9]] : tensor<?x?x?x1x?xf32>
func.func @torch.aten.unsqueeze.dim2.from_end(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,1,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.unsqueeze %arg0, %int-2 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,?,1,?],f32>
}

