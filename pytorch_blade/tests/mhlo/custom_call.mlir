// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_custom_call_ral_qgemm_bias_s8s8s8_perchannel
func.func @torch_blade_custom_call_ral_qgemm_bias_s8s8s8_perchannel(%arg0: !torch.vtensor<[?,?,?],si8>, %weight: !torch.vtensor<[128,128],si8>, %bias: !torch.vtensor<[128],f32>) -> !torch.vtensor<[?,?,128],si8> {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<0.111> : tensor<f32>) : !torch.vtensor<[],f32>
  %1 = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %2 = torch.vtensor.literal(dense_resource<__elided__> : tensor<128xf32>) : !torch.vtensor<[128],f32>
  %3 = torch.vtensor.literal(dense<0> : tensor<128xsi32>) : !torch.vtensor<[128],si32>
  %4 = torch.vtensor.literal(dense<0.222> : tensor<f32>) : !torch.vtensor<[],f32>
  %5 = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  // CHECK: mhlo_disc.custom_call_v2
  // CHECK-SAME: call_target_name = "disc.custom_call.ral_qgemm"
  // CHECK-SAME: has_side_effect = false
  %6 = torch.operator "torch_blade.custom_call"(%arg0, %weight, %bias, %0, %1, %2, %3, %4, %5) {call_target_name = "disc.custom_call.ral_qgemm", device = "cpu", expected_input_layouts = "AB", expected_output_layouts = "AB", input_layouts = "AB", input_placements = "cpu,cpu", output_layouts = "AB", output_placements = "cpu"} : (!torch.vtensor<[?,?,?],si8>, !torch.vtensor<[128,128],si8>, !torch.vtensor<[128],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[?,?,128],si8>
  // CHECK: return
  return %6 : !torch.vtensor<[?,?,128],si8>
}
