// RUN: torch-disc-pdll --payload-input %s --pdl-input %p/fake_quant.pdll  | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_custom_fake_quant
func.func @torch_blade_custom_fake_quant(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],f32> {
  %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %scale = torch.vtensor.literal(dense<0.1> : tensor<f32>) : !torch.vtensor<[],f32>
  %zero_point = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  // CHECK: torch.operator "torch_blade.quantize"
  // CHECK: %[[T0:.*]] = torch.operator "torch_blade.dequantize"
  // CHECK: return %[[T0]]
  %result = torch.operator "torch_blade.fake_quant"(%arg0, %scale, %zero_point, %int-128, %int127, %int8, %3, %true, %true, %false, %false) : (!torch.vtensor<[4,8],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}