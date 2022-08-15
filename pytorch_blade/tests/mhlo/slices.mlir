// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.select.int(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x65x256xf32>) -> tensor<4x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.slice"(%[[ARG0]]) {limit_indices = dense<[4, 2, 256]> : tensor<3xi64>, start_indices = dense<[0, 1, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x65x256xf32>) -> tensor<4x1x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.reshape %[[T0]] : (tensor<4x1x256xf32>) -> tensor<4x256xf32>
// CHECK:         return %[[T1]] : tensor<4x256xf32>
func.func @torch.aten.select.int(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,256],f32> {
  %int1 = torch.constant.int 1
  %2 = torch.aten.select.int %arg0, %int1, %int1 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,256],f32>
  return %2 : !torch.vtensor<[4,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.stride2(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x65x256xf32>) -> tensor<2x65x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.slice"(%[[ARG0]]) {limit_indices = dense<[4, 65, 256]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<[2, 1, 1]> : tensor<3xi64>} : (tensor<4x65x256xf32>) -> tensor<2x65x256xf32>
// CHECK:         return %[[T0]] : tensor<2x65x256xf32>
func.func @torch.aten.slice.stride2(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[2,65,256],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int9223372036854775807, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,65,256],f32>
  return %0 : !torch.vtensor<[2,65,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.dim1(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x65x256xf32>) -> tensor<4x1x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.slice"(%[[ARG0]]) {limit_indices = dense<[4, 65, 256]> : tensor<3xi64>, start_indices = dense<[0, 64, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x65x256xf32>) -> tensor<4x1x256xf32>
// CHECK:         return %[[T0]] : tensor<4x1x256xf32>
func.func @torch.aten.slice.dim1(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,1,256],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-1, %int0, %int1 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1,256],f32>
  return %0 : !torch.vtensor<[4,1,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.none(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<4x65x256xf32>) -> tensor<4x33x256xf32> {
// CHECK:         %[[T0:.*]] = "mhlo.slice"(%[[ARG0]]) {limit_indices = dense<[4, 65, 256]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<4x65x256xf32>) -> tensor<4x33x256xf32>
// CHECK:         return %[[T0]] : tensor<4x33x256xf32>
func.func @torch.aten.slice.none(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,33,256],f32>
  return %0 : !torch.vtensor<[4,33,256],f32>
}

