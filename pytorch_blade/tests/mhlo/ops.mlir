// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @builtin_tensor_interop(
func @builtin_tensor_interop(%arg0: tensor<*xf32>, %arg1: tensor<3x?xi8>, %arg2: !torch.vtensor<*,f32>, %arg3: !torch.vtensor<[3,?],si8>) {
  // CHECK: torch_c.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  // CHECK: torch_c.from_builtin_tensor %arg1 : tensor<3x?xi8> -> !torch.vtensor<[3,?],si8>
  %1 = torch_c.from_builtin_tensor %arg1 : tensor<3x?xi8> -> !torch.vtensor<[3,?],si8>
  // CHECK: torch_c.from_builtin_tensor %arg1 : tensor<3x?xi8> -> !torch.vtensor<[3,?],ui8>
  %2 = torch_c.from_builtin_tensor %arg1 : tensor<3x?xi8> -> !torch.vtensor<[3,?],ui8>
  // CHECK: torch_c.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  %3 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  // CHECK: torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xi8>
  %4 = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xi8>
  return
}

// CHECK-LABEL:  func @torch.tensor_static_info_cast(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:         return %[[ARG0]] : tensor<?x?x?x?xf32>
func @torch.tensor_static_info_cast(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[2,3,224,224],f32> {
  %0 = torch.tensor_static_info_cast %arg0: !torch.vtensor<[?,?,?,?],f32> to !torch.vtensor<[2,3,224,224],f32>
  return %0 : !torch.vtensor<[2,3,224,224],f32>
}
