// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func @torch.tensor_static_info_cast(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:         return %[[ARG0]] : tensor<?x?x?x?xf32>
func @torch.tensor_static_info_cast(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[2,3,224,224],f32> {
  %0 = torch.tensor_static_info_cast %arg0: !torch.vtensor<[?,?,?,?],f32> to !torch.vtensor<[2,3,224,224],f32>
  return %0 : !torch.vtensor<[2,3,224,224],f32>
}
