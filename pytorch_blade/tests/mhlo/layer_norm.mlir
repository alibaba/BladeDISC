// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.layer_norm(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T4:.*]] = mhlo.reshape %[[T3]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T5:.*]] = chlo.broadcast_divide %[[T4]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T6:.*]] = "mhlo.broadcast_in_dim"(%[[T5]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T7:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T6]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_multiply %[[T7]], %[[T7]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = mhlo.reduce(%[[T8]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T10:.*]] = mhlo.reshape %[[T9]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_divide %[[T10]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T12:.*]] = chlo.broadcast_add %[[T11]], %[[T0]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = mhlo.rsqrt %[[T12]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = "mhlo.broadcast_in_dim"(%[[T13]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_multiply %[[T7]], %[[T14]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T15]] : tensor<2x3x224x224xf32>
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
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T5:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T6:.*]] = mhlo.reshape %[[T5]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T7:.*]] = chlo.broadcast_divide %[[T6]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T8:.*]] = "mhlo.broadcast_in_dim"(%[[T7]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T8]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T9]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = mhlo.reduce(%[[T10]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T12:.*]] = mhlo.reshape %[[T11]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = chlo.broadcast_divide %[[T12]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_add %[[T13]], %[[T0]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T15:.*]] = mhlo.rsqrt %[[T14]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = "mhlo.broadcast_in_dim"(%[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_multiply %[[T9]], %[[T16]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T17]], %[[T3]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T19:.*]] = chlo.broadcast_add %[[T18]], %[[T4]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T19]] : tensor<2x3x224x224xf32>
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
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T1]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i32
// CHECK:         %[[T5:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i32
// CHECK:         %[[T7:.*]] = tensor.from_elements %[[T4]], %[[T6]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T8:.*]] = mhlo.dynamic_reshape %[[T2]], %[[T7]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T9:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : index to i64
// CHECK:         %[[T11:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T11]] : index to i64
// CHECK:         %[[T13:.*]] = arith.muli %[[T10]], %[[T12]] : i64
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T13]] : tensor<1xi64>
// CHECK:         %[[T15:.*]] = mhlo.convert(%[[T14]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T16:.*]] = mhlo.reshape %[[T15]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_divide %[[T8]], %[[T16]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[T20:.*]] = arith.trunci %[[T18]] : i64 to i32
// CHECK:         %[[T21:.*]] = arith.trunci %[[T19]] : i64 to i32
// CHECK:         %[[T22:.*]] = arith.trunci %[[T10]] : i64 to i32
// CHECK:         %[[T23:.*]] = arith.trunci %[[T12]] : i64 to i32
// CHECK:         %[[T24:.*]] = tensor.from_elements %[[T20]], %[[T21]], %[[T22]], %[[T23]] : tensor<4xi32>
// CHECK:         %[[T25:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T17]], %[[T24]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T26:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T25]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T27:.*]] = chlo.broadcast_multiply %[[T26]], %[[T26]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T28:.*]] = mhlo.reduce(%[[T27]] init: %[[T1]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T29:.*]] = tensor.dim %[[T27]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T30:.*]] = arith.index_cast %[[T29]] : index to i32
// CHECK:         %[[T31:.*]] = tensor.dim %[[T27]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T32:.*]] = arith.index_cast %[[T31]] : index to i32
// CHECK:         %[[T33:.*]] = tensor.from_elements %[[T30]], %[[T32]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T34:.*]] = mhlo.dynamic_reshape %[[T28]], %[[T33]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T35:.*]] = tensor.dim %[[T27]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T36:.*]] = arith.index_cast %[[T35]] : index to i64
// CHECK:         %[[T37:.*]] = tensor.dim %[[T27]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T38:.*]] = arith.index_cast %[[T37]] : index to i64
// CHECK:         %[[T39:.*]] = arith.muli %[[T36]], %[[T38]] : i64
// CHECK:         %[[T40:.*]] = tensor.from_elements %[[T39]] : tensor<1xi64>
// CHECK:         %[[T41:.*]] = mhlo.convert(%[[T40]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T42:.*]] = mhlo.reshape %[[T41]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T43:.*]] = chlo.broadcast_divide %[[T34]], %[[T42]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T44:.*]] = chlo.broadcast_add %[[T43]], %[[T0]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T45:.*]] = mhlo.rsqrt %[[T44]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T46:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T45]], %[[T24]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T47:.*]] = chlo.broadcast_multiply %[[T26]], %[[T46]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         return %[[T47]] : tensor<?x?x?x?xf32>
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
// CHECK:         %[[C224_I32:.*]] = arith.constant 224 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T5]], %[[T7]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[T3]], %[[T8]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_divide %[[T9]], %[[T1]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T13:.*]] = arith.trunci %[[T11]] : i64 to i32
// CHECK:         %[[T14:.*]] = arith.trunci %[[T12]] : i64 to i32
// CHECK:         %[[T15:.*]] = tensor.from_elements %[[T13]], %[[T14]], %[[C224_I32]], %[[C224_I32]] : tensor<4xi32>
// CHECK:         %[[T16:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T10]], %[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T16]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T17]], %[[T17]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T19:.*]] = mhlo.reduce(%[[T18]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T20:.*]] = tensor.dim %[[T18]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i32
// CHECK:         %[[T22:.*]] = tensor.dim %[[T18]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i32
// CHECK:         %[[T24:.*]] = tensor.from_elements %[[T21]], %[[T23]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T25:.*]] = mhlo.dynamic_reshape %[[T19]], %[[T24]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T26:.*]] = chlo.broadcast_divide %[[T25]], %[[T1]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T27:.*]] = chlo.broadcast_add %[[T26]], %[[T0]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T28:.*]] = mhlo.rsqrt %[[T27]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T29:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T28]], %[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T30:.*]] = chlo.broadcast_multiply %[[T17]], %[[T29]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         return %[[T30]] : tensor<?x?x224x224xf32>
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
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T4:.*]] = mhlo.reshape %[[T3]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T5:.*]] = chlo.broadcast_divide %[[T4]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T6:.*]] = "mhlo.broadcast_in_dim"(%[[T5]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T7:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T6]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T8:.*]] = chlo.broadcast_multiply %[[T7]], %[[T7]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = mhlo.reduce(%[[T8]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T10:.*]] = mhlo.reshape %[[T9]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T11:.*]] = chlo.broadcast_divide %[[T10]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T12:.*]] = chlo.broadcast_add %[[T11]], %[[T0]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = mhlo.rsqrt %[[T12]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = "mhlo.broadcast_in_dim"(%[[T13]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T15:.*]] = chlo.broadcast_multiply %[[T7]], %[[T14]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T15]], %[[T5]], %[[T13]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
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
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0.000000e+00> : tensor<224x224xf32>
// CHECK:         %[[T5:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T6:.*]] = mhlo.reshape %[[T5]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T7:.*]] = chlo.broadcast_divide %[[T6]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T8:.*]] = "mhlo.broadcast_in_dim"(%[[T7]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T9:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T8]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_multiply %[[T9]], %[[T9]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T11:.*]] = mhlo.reduce(%[[T10]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<2x3x224x224xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:         %[[T12:.*]] = mhlo.reshape %[[T11]] : (tensor<2x3xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = chlo.broadcast_divide %[[T12]], %[[T1]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T14:.*]] = chlo.broadcast_add %[[T13]], %[[T0]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T15:.*]] = mhlo.rsqrt %[[T14]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T16:.*]] = "mhlo.broadcast_in_dim"(%[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_multiply %[[T9]], %[[T16]] : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T17]], %[[T3]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         %[[T19:.*]] = chlo.broadcast_add %[[T18]], %[[T4]] : (tensor<2x3x224x224xf32>, tensor<224x224xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T19]], %[[T7]], %[[T15]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
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
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T1]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.reshape %[[T2]] : (tensor<?x?xf32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = arith.muli %[[T5]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T8]] : tensor<1xi64>
// CHECK:         %[[T10:.*]] = mhlo.convert(%[[T9]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T11:.*]] = mhlo.reshape %[[T10]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T12:.*]] = chlo.broadcast_divide %[[T3]], %[[T11]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T13:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T13]] : index to i64
// CHECK:         %[[T15:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T16:.*]] = arith.index_cast %[[T15]] : index to i64
// CHECK:         %[[T17:.*]] = arith.trunci %[[T14]] : i64 to i32
// CHECK:         %[[T18:.*]] = arith.trunci %[[T16]] : i64 to i32
// CHECK:         %[[T19:.*]] = arith.trunci %[[T5]] : i64 to i32
// CHECK:         %[[T20:.*]] = arith.trunci %[[T7]] : i64 to i32
// CHECK:         %[[T21:.*]] = tensor.from_elements %[[T17]], %[[T18]], %[[T19]], %[[T20]] : tensor<4xi32>
// CHECK:         %[[T22:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T12]], %[[T21]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T23:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T22]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T24:.*]] = chlo.broadcast_multiply %[[T23]], %[[T23]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T25:.*]] = mhlo.reduce(%[[T24]] init: %[[T1]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
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
// CHECK:         %[[T36:.*]] = chlo.broadcast_add %[[T35]], %[[T0]] : (tensor<2x3x1x1xf32>, tensor<f32>) -> tensor<2x3x1x1xf32>
// CHECK:         %[[T37:.*]] = mhlo.rsqrt %[[T36]] : tensor<2x3x1x1xf32>
// CHECK:         %[[T38:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T37]], %[[T21]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x3x1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T39:.*]] = chlo.broadcast_multiply %[[T23]], %[[T38]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[T40:.*]] = mhlo.convert(%[[T39]]) : (tensor<?x?x?x?xf32>) -> tensor<2x3x224x224xf32>
// CHECK:         return %[[T40]], %[[T12]], %[[T37]] : tensor<2x3x224x224xf32>, tensor<2x3x1x1xf32>, tensor<2x3x1x1xf32>
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
// CHECK:         %[[C224_I32:.*]] = arith.constant 224 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<9.99999974E-6> : tensor<f32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<5.017600e+04> : tensor<f32>
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i32
// CHECK:         %[[T6:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i32
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T5]], %[[T7]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[T3]], %[[T8]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T10:.*]] = chlo.broadcast_divide %[[T9]], %[[T1]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T13:.*]] = arith.trunci %[[T11]] : i64 to i32
// CHECK:         %[[T14:.*]] = arith.trunci %[[T12]] : i64 to i32
// CHECK:         %[[T15:.*]] = tensor.from_elements %[[T13]], %[[T14]], %[[C224_I32]], %[[C224_I32]] : tensor<4xi32>
// CHECK:         %[[T16:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T10]], %[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T17:.*]] = chlo.broadcast_subtract %[[ARG0]], %[[T16]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T18:.*]] = chlo.broadcast_multiply %[[T17]], %[[T17]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T19:.*]] = mhlo.reduce(%[[T18]] init: %[[T2]]) applies mhlo.add across dimensions = [2, 3] : (tensor<?x?x224x224xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T20:.*]] = tensor.dim %[[T18]], %[[C0]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i32
// CHECK:         %[[T22:.*]] = tensor.dim %[[T18]], %[[C1]] : tensor<?x?x224x224xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i32
// CHECK:         %[[T24:.*]] = tensor.from_elements %[[T21]], %[[T23]], %[[C1_I32]], %[[C1_I32]] : tensor<4xi32>
// CHECK:         %[[T25:.*]] = mhlo.dynamic_reshape %[[T19]], %[[T24]] : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T26:.*]] = chlo.broadcast_divide %[[T25]], %[[T1]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T27:.*]] = chlo.broadcast_add %[[T26]], %[[T0]] : (tensor<?x?x1x1xf32>, tensor<f32>) -> tensor<?x?x1x1xf32>
// CHECK:         %[[T28:.*]] = mhlo.rsqrt %[[T27]] : tensor<?x?x1x1xf32>
// CHECK:         %[[T29:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T28]], %[[T15]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<?x?x1x1xf32>, tensor<4xi32>) -> tensor<?x?x224x224xf32>
// CHECK:         %[[T30:.*]] = chlo.broadcast_multiply %[[T17]], %[[T29]] : (tensor<?x?x224x224xf32>, tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
// CHECK:         return %[[T30]], %[[T10]], %[[T28]] : tensor<?x?x224x224xf32>, tensor<?x?x1x1xf32>, tensor<?x?x1x1xf32>
func.func @torch.aten.native_layer_norm.dynamic_shape.partial(%arg0: !torch.vtensor<[?,?,224,224],f32>) -> (!torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>) {
  %int224 = torch.constant.int 224
  %none = torch.constant.none
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %0 = torch.prim.ListConstruct %int224, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1, %2, %3 = torch.aten.native_layer_norm %arg0, %0, %none, %none, %float1.000000e-05 : !torch.vtensor<[?,?,224,224],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float -> !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>
  return %1, %2, %3 : !torch.vtensor<[?,?,224,224],f32>, !torch.vtensor<[?,?,1,1],f32>, !torch.vtensor<[?,?,1,1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.native_layer_norm_backward.dynamic_shape.partial(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x1xf32>, %[[ARG3:.*]]: tensor<?x?x1xf32>, %[[ARG4:.*]]: tensor<768xf32>, %[[ARG5:.*]]: tensor<768xf32>) -> (tensor<?x?x?xf32>, tensor<768xf32>, tensor<768xf32>) {
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T1:.*]] = chlo.broadcast_subtract %[[ARG1]], %[[ARG2]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T2:.*]] = chlo.broadcast_multiply %[[T1]], %[[ARG3]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T3:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[ARG4]] : (tensor<?x?x?xf32>, tensor<768xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = chlo.broadcast_multiply %[[ARG0]], %[[T2]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = mhlo.reduce(%[[T4]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T6:.*]] = tensor.cast %[[T5]] : tensor<?xf32> to tensor<768xf32>
// CHECK:         %[[T7:.*]] = mhlo.reduce(%[[ARG0]] init: %[[T0]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T8:.*]] = tensor.cast %[[T7]] : tensor<?xf32> to tensor<768xf32>
// CHECK:         %[[T9:.*]] = mhlo.reduce(%[[T3]] init: %[[T0]]) applies mhlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = tensor.dim %[[T3]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i32
// CHECK:         %[[T12:.*]] = tensor.dim %[[T3]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T13:.*]] = arith.index_cast %[[T12]] : index to i32
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T11]], %[[T13]], %[[C1_I32]] : tensor<3xi32>
// CHECK:         %[[T15:.*]] = mhlo.dynamic_reshape %[[T9]], %[[T14]] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T16:.*]] = tensor.dim %[[T3]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[T16]] : index to i64
// CHECK:         %[[T18:.*]] = tensor.from_elements %[[T17]] : tensor<1xi64>
// CHECK:         %[[T19:.*]] = mhlo.convert(%[[T18]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T20:.*]] = mhlo.reshape %[[T19]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T21:.*]] = chlo.broadcast_divide %[[T15]], %[[T20]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T22:.*]] = chlo.broadcast_multiply %[[T3]], %[[T2]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T23:.*]] = mhlo.reduce(%[[T22]] init: %[[T0]]) applies mhlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK:         %[[T24:.*]] = tensor.dim %[[T22]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T25:.*]] = arith.index_cast %[[T24]] : index to i32
// CHECK:         %[[T26:.*]] = tensor.dim %[[T22]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T27:.*]] = arith.index_cast %[[T26]] : index to i32
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[T25]], %[[T27]], %[[C1_I32]] : tensor<3xi32>
// CHECK:         %[[T29:.*]] = mhlo.dynamic_reshape %[[T23]], %[[T28]] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T30:.*]] = tensor.dim %[[T22]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T31:.*]] = arith.index_cast %[[T30]] : index to i64
// CHECK:         %[[T32:.*]] = tensor.from_elements %[[T31]] : tensor<1xi64>
// CHECK:         %[[T33:.*]] = mhlo.convert(%[[T32]]) : (tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T34:.*]] = mhlo.reshape %[[T33]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T35:.*]] = chlo.broadcast_divide %[[T29]], %[[T34]] : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
// CHECK:         %[[T36:.*]] = chlo.broadcast_multiply %[[T2]], %[[T35]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T37:.*]] = chlo.broadcast_subtract %[[T3]], %[[T21]] : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T38:.*]] = chlo.broadcast_subtract %[[T37]], %[[T36]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T39:.*]] = chlo.broadcast_multiply %[[ARG3]], %[[T38]] : (tensor<?x?x1xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         return %[[T39]], %[[T6]], %[[T8]] : tensor<?x?x?xf32>, tensor<768xf32>, tensor<768xf32>
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

