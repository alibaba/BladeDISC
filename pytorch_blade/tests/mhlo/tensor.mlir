// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK:     %cst = arith.constant dense<5> : tensor<i64>
// CHECK:     return %cst
func @torch.aten.tensor.int() -> !torch.tensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int5 = torch.constant.int 5
  %0 = torch.aten.tensor.int %int5, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK: %cst = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK: return %cst
func @main() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi64>) : !torch.tensor<[4],si64>
  %1 = torch.tensor_static_info_cast %0 : !torch.tensor<[4],si64> to !torch.tensor
  return %1 : !torch.tensor
}