// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.layer_norm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T5:.*]] = mhlo.reshape %[[T4]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T6:.*]] = chlo.broadcast_divide %[[T5]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T7:.*]] = "mhlo.broadcast_in_dim"(%[[T6]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_multiply %[[T7]], %[[T1]] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T8]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T9]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = mhlo.reduce(%[[T10]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T12:.*]] = mhlo.reshape %[[T11]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = chlo.broadcast_divide %[[T12]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_add %[[T13]], %[[T14]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = mhlo.rsqrt %[[T15]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T17:.*]] = "mhlo.broadcast_in_dim"(%[[T16]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T9]], %[[T17]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T18]] : tensor<2x3x224x224xf32>
func.func @torch.aten.layer_norm(%arg0: !torch.vtensor<[2,3,224,224],f32>) -> !torch.vtensor<[2,3,224,224],f32> {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.layer_norm %arg0, %0, %none, %none, %float1.000000e-05, %true : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float, !torch.bool -> !torch.vtensor<[2,3,224,224],f32>
  return %1 : !torch.vtensor<[2,3,224,224],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.layer_norm.affine(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<1.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T5:.*]] = mhlo.constant dense<0.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T6:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T7:.*]] = mhlo.reshape %[[T6]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_divide %[[T7]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T9:.*]] = "mhlo.broadcast_in_dim"(%[[T8]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T1]] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T10]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T12:.*]] = chlo.broadcast_multiply %[[T11]], %[[T11]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T13:.*]] = mhlo.reduce(%[[T12]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T14:.*]] = mhlo.reshape %[[T13]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_divide %[[T14]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_add %[[T15]], %[[T16]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T18:.*]] = mhlo.rsqrt %[[T17]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T19:.*]] = "mhlo.broadcast_in_dim"(%[[T18]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T20:.*]] = chlo.broadcast_multiply %[[T11]], %[[T19]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T21:.*]] = chlo.broadcast_multiply %[[T20]], %[[T4]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T22:.*]] = chlo.broadcast_multiply %[[T5]], %[[T1]] : (tensor<224x224xf32>, tensor<f32>) -> tensor<224x224xf32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_add %[[T21]], %[[T22]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T23]] : tensor<2x3x224x224xf32>
func.func @torch.aten.layer_norm.affine(%arg0: !torch.vtensor<[2,3,224,224],f32>) -> !torch.vtensor<[2,3,224,224],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<224x224xf32>) : !torch.vtensor<[224,224],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<224x224xf32>) : !torch.vtensor<[224,224],f32>
  %int224 = torch.constant.int 224
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.layer_norm %arg0, %2, %1, %0, %float1.000000e-05, %true : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int>, !torch.vtensor<[224,224],f32>, !torch.vtensor<[224,224],f32>, !torch.float, !torch.bool -> !torch.vtensor<[2,3,224,224],f32>
  return %3 : !torch.vtensor<[2,3,224,224],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.layer_norm.dynamic_shape.full(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T5]], %[[T7]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[T3]], %[[T8]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T10:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i64
// CHECK:         %[[T12:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T13:.*]] = arith.index_cast %[[T12]] : index to i64
// CHECK:         %[[T14:.*]] = arith.muli %[[T11]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = tensor.from_elements %[[T14]] : tensor<1xi64>
// CHECK:         %[[T16:.*]] = mhlo.convert(%[[T15]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T17:.*]] = mhlo.reshape %[[T16]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_divide %[[T9]], %[[T17]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T20:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:         %[[T21:.*]] = tensor.from_elements %[[T5]], %[[T7]], %[[T19]], %[[T20]] : tensor<4xi32>
// CHECK:         %[[T22:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T18]], %[[T21]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_multiply %[[T22]], %[[T1]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T23]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T25:.*]] = chlo.broadcast_multiply %[[T24]], %[[T24]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T26:.*]] = mhlo.reduce(%[[T25]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T27:.*]] = tensor.dim %[[T25]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T28:.*]] = arith.index_cast %[[T27]] : index to i32
// CHECK:         %[[T29:.*]] = tensor.dim %[[T25]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T30:.*]] = arith.index_cast %[[T29]] : index to i32
// CHECK:         %[[T31:.*]] = tensor.from_elements %[[T28]], %[[T30]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T32:.*]] = mhlo.dynamic_reshape %[[T26]], %[[T31]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T33:.*]] = tensor.dim %[[T25]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T34:.*]] = arith.index_cast %[[T33]] : index to i64
// CHECK:         %[[T35:.*]] = tensor.dim %[[T25]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T36:.*]] = arith.index_cast %[[T35]] : index to i64
// CHECK:         %[[T37:.*]] = arith.muli %[[T34]], %[[T36]] : i64
// CHECK:         %[[T38:.*]] = tensor.from_elements %[[T37]] : tensor<1xi64>
// CHECK:         %[[T39:.*]] = mhlo.convert(%[[T38]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T40:.*]] = mhlo.reshape %[[T39]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T41:.*]] = chlo.broadcast_divide %[[T32]], %[[T40]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T42:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T43:.*]] = chlo.broadcast_add %[[T41]], %[[T42]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T44:.*]] = mhlo.rsqrt %[[T43]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T45:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T44]], %[[T21]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T46:.*]] = chlo.broadcast_multiply %[[T24]], %[[T45]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         return %[[T46]] : tensor<?x?x?x?xf32>
func.func @torch.aten.layer_norm.dynamic_shape.full(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.layer_norm %arg0, %0, %none, %none, %float1.000000e-05, %true : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %1 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.layer_norm.dynamic_shape.partial(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[C224_I32:.*]] = arith.constant 224 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i32
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T6]], %[[T8]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T4]], %[[T9]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_divide %[[T10]], %[[T2]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T6]], %[[T8]], %[[C224_I32]], %[[C224_I32]] : tensor<4xi32>
// CHECK:         %[[T13:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T11]], %[[T12]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_multiply %[[T13]], %[[T1]] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T14]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T16:.*]] = chlo.broadcast_multiply %[[T15]], %[[T15]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T17:.*]] = mhlo.reduce(%[[T16]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T18:.*]] = tensor.dim %[[T16]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i32
// CHECK:         %[[T20:.*]] = tensor.dim %[[T16]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i32
// CHECK:         %[[T22:.*]] = tensor.from_elements %[[T19]], %[[T21]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T23:.*]] = mhlo.dynamic_reshape %[[T17]], %[[T22]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_divide %[[T23]], %[[T2]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T25:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T26:.*]] = chlo.broadcast_add %[[T24]], %[[T25]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T27:.*]] = mhlo.rsqrt %[[T26]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T28:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T27]], %[[T12]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T29:.*]] = chlo.broadcast_multiply %[[T15]], %[[T28]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         return %[[T29]] : tensor<?x?x224x224xf32>
func.func @torch.aten.layer_norm.dynamic_shape.partial(%arg0: !torch.vtensor<[?,?,224,224],f32>) -> !torch.vtensor<[?,?,224,224],f32> {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.layer_norm %arg0, %0, %none, %none, %float1.000000e-05, %true : !torch.vtensor<[?,?,224,224],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float, !torch.bool -> !torch.vtensor<[?,?,224,224],f32>
  return %1 : !torch.vtensor<[?,?,224,224],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_layer_norm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> (tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T5:.*]] = mhlo.reshape %[[T4]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T6:.*]] = chlo.broadcast_divide %[[T5]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T7:.*]] = "mhlo.broadcast_in_dim"(%[[T6]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_multiply %[[T7]], %[[T1]] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T8]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T9]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = mhlo.reduce(%[[T10]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T12:.*]] = mhlo.reshape %[[T11]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = chlo.broadcast_divide %[[T12]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_add %[[T13]], %[[T14]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = mhlo.rsqrt %[[T15]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T17:.*]] = "mhlo.broadcast_in_dim"(%[[T16]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T9]], %[[T17]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T18]], %[[T6]], %[[T16]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
func.func @torch.aten.native_layer_norm(%arg0: !torch.vtensor<[2,3,224,224],f32>) -> (!torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>) {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1, %2, %3 = torch.aten.native_layer_norm %arg0, %0, %none, %none, %float1.000000e-05 : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float -> !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
  return %1, %2, %3 : !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_layer_norm.affine(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> (tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<1.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T5:.*]] = mhlo.constant dense<0.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T6:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T7:.*]] = mhlo.reshape %[[T6]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_divide %[[T7]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T9:.*]] = "mhlo.broadcast_in_dim"(%[[T8]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T1]] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T10]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T12:.*]] = chlo.broadcast_multiply %[[T11]], %[[T11]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T13:.*]] = mhlo.reduce(%[[T12]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T14:.*]] = mhlo.reshape %[[T13]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_divide %[[T14]], %[[T2]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_add %[[T15]], %[[T16]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T18:.*]] = mhlo.rsqrt %[[T17]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T19:.*]] = "mhlo.broadcast_in_dim"(%[[T18]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T20:.*]] = chlo.broadcast_multiply %[[T11]], %[[T19]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T21:.*]] = chlo.broadcast_multiply %[[T20]], %[[T4]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T22:.*]] = chlo.broadcast_multiply %[[T5]], %[[T1]] : (tensor<224x224xf32>, tensor<f32>) -> tensor<224x224xf32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_add %[[T21]], %[[T22]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T23]], %[[T8]], %[[T18]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
func.func @torch.aten.native_layer_norm.affine(%arg0: !torch.vtensor<[2,3,224,224],f32>) -> (!torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>) {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<224x224xf32>) : !torch.vtensor<[224,224],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<224x224xf32>) : !torch.vtensor<[224,224],f32>
  %int224 = torch.constant.int 224
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %2 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %3, %4, %5 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int>, !torch.vtensor<[224,224],f32>, !torch.vtensor<[224,224],f32>, !torch.float -> !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
  return %3, %4, %5 : !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_layer_norm.dynamic_shape.full(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> (tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = mhlo.reshape %[[T3]] : (tensor<?x?xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[T9:.*]] = arith.muli %[[T6]], %[[T8]] : i64
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T9]] : tensor<1xi64>
// CHECK:         %[[T11:.*]] = mhlo.convert(%[[T10]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T12:.*]] = mhlo.reshape %[[T11]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T13:.*]] = chlo.broadcast_divide %[[T4]], %[[T12]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T15:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T16:.*]] = arith.index_cast %[[T14]] : index to i32
// CHECK:         %[[T17:.*]] = arith.index_cast %[[T15]] : index to i32
// CHECK:         %[[T18:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T7]] : index to i32
// CHECK:         %[[T20:.*]] = tensor.from_elements %[[T16]], %[[T17]], %[[T18]], %[[T19]] : tensor<4xi32>
// CHECK:         %[[T21:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T13]], %[[T20]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T22:.*]] = chlo.broadcast_multiply %[[T21]], %[[T1]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T22]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_multiply %[[T23]], %[[T23]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T25:.*]] = mhlo.reduce(%[[T24]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T26:.*]] = mhlo.reshape %[[T25]] : (tensor<?x?xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T27:.*]] = tensor.dim %[[T24]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T28:.*]] = arith.index_cast %[[T27]] : index to i64
// CHECK:         %[[T29:.*]] = tensor.dim %[[T24]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T30:.*]] = arith.index_cast %[[T29]] : index to i64
// CHECK:         %[[T31:.*]] = arith.muli %[[T28]], %[[T30]] : i64
// CHECK:         %[[T32:.*]] = tensor.from_elements %[[T31]] : tensor<1xi64>
// CHECK:         %[[T33:.*]] = mhlo.convert(%[[T32]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T34:.*]] = mhlo.reshape %[[T33]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T35:.*]] = chlo.broadcast_divide %[[T26]], %[[T34]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T36:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T37:.*]] = chlo.broadcast_add %[[T35]], %[[T36]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T38:.*]] = mhlo.rsqrt %[[T37]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T39:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T38]], %[[T20]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T40:.*]] = chlo.broadcast_multiply %[[T23]], %[[T39]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T41:.*]] = mhlo.convert(%[[T40]]) : (tensor<?x?x?x?xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T41]], %[[T13]], %[[T38]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
func.func @torch.aten.native_layer_norm.dynamic_shape.full(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> (!torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>) {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1, %2, %3 = torch.aten.native_layer_norm %arg0, %0, %none, %none, %float1.000000e-05 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float -> !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
  return %1, %2, %3 : !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[2,3,1,1],f32>, !torch.vtensor<[2,3,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_layer_norm.dynamic_shape.partial(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x224x224xf32>) -> (tensor<?x?x224x224xf32>, tensor<?x?x1x1xf32>, tensor<?x?x1x1xf32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[C224_I32:.*]] = arith.constant 224 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i32
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T6]], %[[T8]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T4]], %[[T9]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_divide %[[T10]], %[[T2]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T6]], %[[T8]], %[[C224_I32]], %[[C224_I32]] : tensor<4xi32>
// CHECK:         %[[T13:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T11]], %[[T12]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_multiply %[[T13]], %[[T1]] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T14]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T16:.*]] = chlo.broadcast_multiply %[[T15]], %[[T15]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T17:.*]] = mhlo.reduce(%[[T16]] init: %[[T3]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T18:.*]] = tensor.dim %[[T16]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i32
// CHECK:         %[[T20:.*]] = tensor.dim %[[T16]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i32
// CHECK:         %[[T22:.*]] = tensor.from_elements %[[T19]], %[[T21]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T23:.*]] = mhlo.dynamic_reshape %[[T17]], %[[T22]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_divide %[[T23]], %[[T2]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T25:.*]] = chlo.broadcast_multiply %[[T0]], %[[T1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T26:.*]] = chlo.broadcast_add %[[T24]], %[[T25]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T27:.*]] = mhlo.rsqrt %[[T26]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T28:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T27]], %[[T12]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T29:.*]] = chlo.broadcast_multiply %[[T15]], %[[T28]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         return %[[T29]], %[[T11]], %[[T27]] : tensor<?x?x224x224xf32>, tensor<?x?x1x1xf32>, tensor<?x?x1x1xf32>
func.func @torch.aten.native_layer_norm.dynamic_shape.partial(%arg0: !torch.vtensor<[?,?,224,224],f32>) -> (!torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>) {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1, %2, %3 = torch.aten.native_layer_norm %arg0, %0, %none, %none, %float1.000000e-05 : !torch.vtensor<[?,?,224,224],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float -> !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>
  return %1, %2, %3 : !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>
}

// CHECK-LABEL: func.func @torch.aten.native_layer_norm_backward.dynamic_shape.partial(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x1xf32>, %[[ARG3:.*]]: tensor<?x?x1xf32>, %[[ARG4:.*]]: tensor<768xf32>, %[[ARG5:.*]]: tensor<768xf32>) -> (tensor<?x?x?xf32>, tensor<768xf32>, tensor<768xf32>) {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:         %[[CST_2:.*]] = arith.constant 2 : index
// CHECK:         %[[CST_1_I32:.*]]= arith.constant 1 : i32
// CHECK:         %[[CST_1:.*]] = arith.constant 1 : index
// CHECK:         %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = chlo.broadcast_multiply %[[ARG2]], %[[T0]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T3:.*]] = chlo.broadcast_subtract %[[ARG1]], %[[T2]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = chlo.broadcast_multiply %[[T3]], %[[ARG3]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[ARG4]] : (tensor<?x?x?xf32>, tensor<768xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T6:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[T4]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T7:.*]] = mhlo.reduce(%[[T6]] init: %[[T1]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T8:.*]] = mhlo.convert(%[[T7]]) : (tensor<?xf32>) -> tensor<768xf32>
// CHECK:         %[[T9:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T1]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T10:.*]] = mhlo.convert(%[[T9]]) : (tensor<?xf32>) -> tensor<768xf32>
// CHECK:         %[[T11:.*]] = mhlo.reduce(%[[T5]] init: %[[T1]]) applies mhlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T12:.*]] = tensor.dim %[[T5]], %[[CST_0]] : tensor<?x?x?xf32>
// CHECK:         %[[T13:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:         %[[T14:.*]] = tensor.dim %[[T5]], %[[CST_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T15:.*]] = arith.index_cast %[[T14]] : index to i32
// CHECK:         %[[T16:.*]] = tensor.from_elements %[[T13]], %[[T15]], %[[CST_1_I32]]: tensor<3xi32>
// CHECK:         %[[T17:.*]] = "mhlo.dynamic_reshape"(%[[T11]], %[[T16]]) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T18:.*]] = tensor.dim %[[T5]], %[[CST_2]] : tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i64
// CHECK:         %[[T20:.*]] = tensor.from_elements %[[T19]] : tensor<1xi64>
// CHECK:         %[[T21:.*]] = mhlo.convert(%[[T20]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T22:.*]] = "mhlo.reshape"(%[[T21]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_divide %[[T17]], %[[T22]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_multiply %[[T5]], %[[T4]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T25:.*]] = mhlo.reduce(%[[T24]] init: %[[T1]]) applies mhlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T26:.*]] = tensor.dim %[[T24]], %[[CST_0]] : tensor<?x?x?xf32>
// CHECK:         %[[T27:.*]] = arith.index_cast %[[T26]] : index to i32
// CHECK:         %[[T28:.*]] = tensor.dim %[[T24]], %[[CST_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T29:.*]] = arith.index_cast %[[T28]] : index to i32
// CHECK:         %[[T30:.*]] = tensor.from_elements %[[T27]], %[[T29]], %[[CST_1_I32]]: tensor<3xi32>
// CHECK:         %[[T31:.*]] = "mhlo.dynamic_reshape"(%[[T25]], %[[T30]]) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T32:.*]] = tensor.dim %[[T24]], %[[CST_2]] : tensor<?x?x?xf32>
// CHECK:         %[[T33:.*]] = arith.index_cast %[[T32]] : index to i64
// CHECK:         %[[T34:.*]] = tensor.from_elements %[[T33]] : tensor<1xi64>
// CHECK:         %[[T35:.*]] = mhlo.convert(%[[T34]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T36:.*]] = "mhlo.reshape"(%[[T35]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T37:.*]] = chlo.broadcast_divide %[[T31]], %[[T36]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T38:.*]] = chlo.broadcast_multiply %[[T4]], %[[T37]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T39:.*]] = chlo.broadcast_multiply %[[T23]], %[[T0]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T40:.*]] = chlo.broadcast_subtract %[[T5]], %[[T39]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T41:.*]] = chlo.broadcast_multiply %[[T38]], %[[T0]] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T42:.*]] = chlo.broadcast_subtract %[[T40]], %[[T41]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T43:.*]] = chlo.broadcast_multiply %[[ARG3]], %[[T42]] : (tensor<?x?x1xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         return %[[T43]], %[[T8]], %[[T10]] : tensor<?x?x?xf32>, tensor<768xf32>, tensor<768xf32> 
// CHECK:       }
func.func @torch.aten.native_layer_norm_backward.dynamic_shape.partial(
      %arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>, %arg2: !torch.vtensor<[?,?,1],f32>,
      %arg3: !torch.vtensor<[?,?,1],f32>, %arg4: !torch.vtensor<[768],f32>, %arg5: !torch.vtensor<[768],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>) {
  %true = torch.constant.bool true
  %int768 = torch.constant.int 768
  %int_list = torch.prim.ListConstruct %int768 : (!torch.int) -> !torch.list<int>
  %bool_list = torch.prim.ListConstruct %true, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.native_layer_norm_backward %arg0, %arg1, %int_list, %arg2, %arg3, %arg4, %arg5, %bool_list : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>, !torch.list<bool> -> !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>
  return %result0, %result1, %result2 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>
}
