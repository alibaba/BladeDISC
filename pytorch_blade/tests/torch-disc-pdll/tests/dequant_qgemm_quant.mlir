// RUN: torch-disc-pdll --payload-input %s --pdl-input %p/dequant_qgemm_quant.pdll  | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_dequant_qgemm_quant
func.func @torch_blade_dequant_qgemm_quant(%input: !torch.vtensor<[4,8],si8>, %weight: !torch.vtensor<[16,8],si8>, %bias: !torch.vtensor<[16],si32>, %input_s: !torch.vtensor, %input_zp: !torch.vtensor, %weight_s: !torch.vtensor, %weight_zp: !torch.vtensor, %output_s: !torch.vtensor, %output_zp: !torch.vtensor) -> !torch.vtensor<[4,16], si8> {
  %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  // CHECK: %[[T0:.*]] = torch.operator "torch_blade.custom_call"
  // CHECK-SAME: call_target_name = "disc.custom_call.ral_qgemm"
  // CHECK: return %[[T0]]
  %input_dequant = torch.operator "torch_blade.dequantize"(%input, %input_s, %input_zp, %int-128, %int127, %int8, %3, %true, %true, %false, %false) : (!torch.vtensor<[4,8],si8>, !torch.vtensor, !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[4,8],f32>
  %weight_dequant = torch.operator "torch_blade.dequantize"(%weight, %weight_s, %weight_zp, %int-128, %int127, %int8, %3, %true, %true, %false, %false) : (!torch.vtensor<[16,8],si8>, !torch.vtensor, !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[16,8],f32>
  %gemm_output = torch.aten.linear %input_dequant, %weight_dequant, %bias : !torch.vtensor<[4,8],f32>, !torch.vtensor<[16,8],f32>, !torch.vtensor<[16],si32> -> !torch.vtensor<[4,16],f32>
  %result = torch.operator "torch_blade.quantize"(%gemm_output, %output_s, %output_zp, %int-128, %int127, %int8, %3, %true, %true, %false, %false) : (!torch.vtensor<[4,16],f32>, !torch.vtensor, !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[4,16],si8>
  return %result : !torch.vtensor<[4,16],si8>
}