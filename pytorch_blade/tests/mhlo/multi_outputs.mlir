// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_custom_call_multi_outputs
func.func @torch_blade_custom_call_multi_outputs(%arg0: !torch.vtensor<[?,?,?],si8>, %weight: !torch.vtensor<[128,128],si8>, %bias: !torch.vtensor<[128],f32>) -> (!torch.vtensor<[?,?,128],si8>, !torch.vtensor<[?,?,128],si8>) {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %input_scale = torch.vtensor.literal(dense<0.111> : tensor<f32>) : !torch.vtensor<[],f32>
  %input_zero_point = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %weight_scale = torch.vtensor.literal(dense_resource<__elided__> : tensor<128xf32>) : !torch.vtensor<[128],f32>
  %weight_zero_point = torch.vtensor.literal(dense<0> : tensor<128xsi32>) : !torch.vtensor<[128],si32>
  %output_scale = torch.vtensor.literal(dense<0.222> : tensor<f32>) : !torch.vtensor<[],f32>
  %output_zero_point = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  // CHECK: mhlo_disc.custom_call_v2
  // CHECK-SAME: call_target_name = "disc.custom_call.ral_qgemm"
  // CHECK-SAME: custom_attrs = {}
  // CHECK-SAME: device = "h"
  // CHECK-SAME: expected_input_layouts = "ABC,AB,A,,,A,A,,"
  // CHECK-SAME: expected_output_layouts = "ABC,ABC"
  // CHECK-SAME: has_side_effect = false
  // CHECK-SAME: input_layouts = "ABC,AB,A,,,A,A,,"
  // CHECK-SAME: input_placements = "h,h,h,h,h,h,h,h,h"
  // CHECK-SAME: output_layouts = "ABC,ABC"
  // CHECK-SAME: output_placements = "h,h"
  %6:2 = torch.operator "torch_blade.custom_call"(%arg0, %weight, %bias, %input_scale, %input_zero_point, %weight_scale, %weight_zero_point, %output_scale, %output_zero_point) {
        call_target_name = "disc.custom_call.ral_qgemm",
        device = "h",
        expected_input_layouts = "ABC,AB,A,,,A,A,,",
        expected_output_layouts = "ABC,ABC",
        input_layouts = "ABC,AB,A,,,A,A,,",
        input_placements = "h,h,h,h,h,h,h,h,h",
        output_layouts = "ABC,ABC",
        output_placements = "h,h"
  } : (!torch.vtensor<[?,?,?],si8>, !torch.vtensor<[128,128],si8>, !torch.vtensor<[128],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>) -> (!torch.vtensor<[?,?,128],si8>, !torch.vtensor<[?,?,128],si8>)
  // CHECK: return
  return %6#0,%6#1 : !torch.vtensor<[?,?,128],si8>, !torch.vtensor<[?,?,128],si8>
}
