// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.tensor.int(
// CHECK-SAME:         ) -> tensor<i64> {
// CHECK:         %[[CST:.*]] = arith.constant dense<5> : tensor<i64>
// CHECK:         return %[[CST]] : tensor<i64>
func.func @torch.aten.tensor.int() -> !torch.tensor<[], si64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int5 = torch.constant.int 5
  %0 = torch.aten.tensor.int %int5, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.tensor<[],si64>
  return %0 : !torch.tensor<[], si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.literal.sint(
// CHECK-SAME:         ) -> tensor<4xi64> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK:         return %[[T0]] : tensor<4xi64>
func.func @torch.aten.literal.sint() -> !torch.tensor<[4], si64> {
  %0 = torch.tensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi64>) : !torch.tensor<[4],si64>
  return %0 : !torch.tensor<[4], si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.literal.ui8(
// CHECK-SAME:         ) -> tensor<4xui8> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xui8>
// CHECK:         return %[[T0]] : tensor<4xui8>
func.func @torch.aten.literal.ui8() -> !torch.tensor<[4],ui8> {
  %0 = torch.tensor.literal(dense<[1, 2, 3, 4]> : tensor<4xui8>) : !torch.tensor<[4],ui8>
  return %0 : !torch.tensor<[4], ui8>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.literal.ui1(
// CHECK-SAME:         ) -> tensor<4xi1> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<[true, false, false, true]> : tensor<4xi1>
// CHECK:         return %[[T0]] : tensor<4xi1>
func.func @torch.aten.literal.ui1() -> !torch.tensor<[4], i1> {
  %0 = torch.tensor.literal(dense<[1, 0, 0, 1]> : tensor<4xi1>) : !torch.tensor<[4],i1>
  return %0 : !torch.tensor<[4], i1>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.literal.f32(
// CHECK-SAME:         ) -> tensor<4xf32> {
// CHECK:         %[[T0:.*]] = mhlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// CHECK:         return %[[T0]] : tensor<4xf32>
func.func @torch.aten.literal.f32() -> !torch.tensor<[4], f32> {
  %0 = torch.tensor.literal(dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : !torch.tensor<[4], f32>
  return %0 : !torch.tensor<[4], f32>
}

