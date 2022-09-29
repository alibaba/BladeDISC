// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.tensor_static_info_cast(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
// CHECK:         return %[[ARG0]] : tensor<2x3x224x224xf32>
func.func @torch.tensor_static_info_cast(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[2,3,224,224],f32> {
  %0 = torch.tensor_static_info_cast %arg0: !torch.vtensor<[?,?,?,?],f32> to !torch.vtensor<[2,3,224,224],f32>
  return %0 : !torch.vtensor<[2,3,224,224],f32>
}

