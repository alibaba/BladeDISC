// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_custom_fake_quant
func.func @torch_blade_custom_fake_quant(%arg0: !torch.tensor, %arg1: !torch.tensor, %arg2: !torch.tensor) -> !torch.tensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.tensor to !torch.tensor<[2,3],f32>
  %1 = torch.tensor_static_info_cast %arg1 : !torch.tensor to !torch.tensor<[],f32>
  %2 = torch.tensor_static_info_cast %arg2 : !torch.tensor to !torch.tensor<[],si64>
  %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %int-127 = torch.constant.int -127
  %int128 = torch.constant.int 128
  %int8 = torch.constant.int 8
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  // CHECK: mhlo_disc.fake_quant
  // CHECK-SAME: axis = dense<>
  // CHECK-SAME: num_bits = 8
  // CHECK-SAME: quant_max = 128
  // CHECK-SAME: quant_min = -127
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_signed = true
  // CHECK-SAME: use_symmetric = true
  %4 = torch.operator "torch_blade.fake_quant"(%0, %1, %2, %int-127, %int128, %int8, %3, %true, %true, %false, %false) : (!torch.tensor<[2,3],f32>, !torch.tensor<[],f32>, !torch.tensor<[],si64>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.tensor<[2,3],f32>
  %5 = torch.tensor_static_info_cast %4 : !torch.tensor<[2,3],f32> to !torch.tensor
  // CHECK: return
  return %5 : !torch.tensor
}