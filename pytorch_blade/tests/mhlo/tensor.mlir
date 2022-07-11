// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK:   %cst = arith.constant dense<5> : tensor<i64>
// CHECK:   return %cst
func @torch.aten.tensor.int() -> !torch.tensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int5 = torch.constant.int 5
  %0 = torch.aten.tensor.int %int5, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK:   %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK:   return %0
func @torch.aten.literal.sint() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi64>) : !torch.tensor<[4],si64>
  %1 = torch.tensor_static_info_cast %0 : !torch.tensor<[4],si64> to !torch.tensor
  return %1 : !torch.tensor
}


// CHECK:   %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xui8>
// CHECK:   return %0
func @torch.aten.literal.ui8() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<[1, 2, 3, 4]> : tensor<4xui8>) : !torch.tensor<[4],ui8>
  %1 = torch.tensor_static_info_cast %0 : !torch.tensor<[4],ui8> to !torch.tensor
  return %1 : !torch.tensor
}

// CHECK:   %0 = mhlo.constant dense<[true, false, false, true]> : tensor<4xi1>
// CHECK:   return %0
func @torch.aten.literal.ui1() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<[1, 0, 0, 1]> : tensor<4xi1>) : !torch.tensor<[4],i1>
  %1 = torch.tensor_static_info_cast %0 : !torch.tensor<[4], i1> to !torch.tensor
  return %1 : !torch.tensor
}

// CHECK:   %0 = mhlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// CHECK:   return %0
func @torch.aten.literal.f32() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : !torch.tensor<[4], f32>
  %1 = torch.tensor_static_info_cast %0 : !torch.tensor<[4], f32> to !torch.tensor
  return %1 : !torch.tensor
}