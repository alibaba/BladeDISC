// RUN: torch-disc-pdll --payload-input %s --pdl-input %p/fake_quant.pdll  | FileCheck %s

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
  // CHECK: torch.operator
  %result = torch.operator "torch_blade.fake_quant"(%0, %1, %2, %int-127, %int128, %int8, %3, %true, %true, %false, %false) : (!torch.tensor<[2,3],f32>, !torch.tensor<[],f32>, !torch.tensor<[],si64>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.tensor<[2,3],f32>
  %5 = torch.tensor_static_info_cast %result : !torch.tensor<[2,3],f32> to !torch.tensor
  return %5 : !torch.tensor
}