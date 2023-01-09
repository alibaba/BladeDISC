// RUN: disc-opt -disc-quantized-dot-merge -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @quantized_dot_merging_dynamic_three
func.func @quantized_dot_merging_dynamic_three(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x768xf32> {  
  %10 = mhlo.constant dense<0> : tensor<768xi8>
  %130 = mhlo.constant dense_resource<__elided__> : tensor<768xi8>
  %131 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>
  %132 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>
  %133 = mhlo.constant dense_resource<__elided__> : tensor<768xi8>
  %134 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>

  %143 = mhlo.constant dense<0.0741644949> : tensor<f32>
  %144 = mhlo.constant dense<0> : tensor<i32>
  %145 = mhlo.constant dense<0.0022647616> : tensor<f32>
  %146 = mhlo.constant dense<0.0447170027> : tensor<f32>
  %150 = mhlo.constant dense<0.0264002532> : tensor<f32>

  // CHECK: %[[C0:.*]] = "mhlo.concatenate"
  // CHECK: %[[T0:.*]] = "mhlo.broadcast_in_dim"
  // CHECK: %[[C1:.*]] = "mhlo.concatenate"
  // CHECK: %[[C2:.*]] = "mhlo.concatenate"
  // CHECK: %[[C3:.*]] = "mhlo.concatenate"
  // CHECK: %[[C4:.*]] = "mhlo.concatenate"
  // CHECK: %[[C5:.*]] = "mhlo.concatenate"
  // CHECK: %[[C6:.*]] = "mhlo.concatenate"
  // CHECK: %[[C7:.*]] = "mhlo.concatenate"
  // CHECK: %[[T2:.*]] = "mhlo_disc.custom_call_v2"
  // CHECK: %[[T3:.*]] = mhlo.real_dynamic_slice %[[T2]]
  // CHECK: %[[T4:.*]] = mhlo.real_dynamic_slice %[[T2]]
  // CHECK: %[[T5:.*]] = mhlo.real_dynamic_slice %[[T2]]
  // CHECK: %[[T6:.*]] = "mhlo_disc.dequantize"
  %427 = "mhlo_disc.quantize"(%arg0, %143, %144) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?xi8>
  %428 = "mhlo_disc.custom_call_v2"(%427, %134, %133, %143, %144, %145, %144, %146, %144) {call_target_name = "ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,d,s,d,s,d,s", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xi8>, tensor<768x768xi8>, tensor<768xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xi8>
  %430 = "mhlo_disc.custom_call_v2"(%427, %132, %10, %143, %144, %145, %144, %146, %144) {call_target_name = "ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,d,s,d,s,d,s", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xi8>, tensor<768x768xi8>, tensor<768xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xi8>
  %443 = "mhlo_disc.custom_call_v2"(%427, %131, %130, %143, %144, %145, %144, %146, %144) {call_target_name = "ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,d,s,d,s,d,s", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xi8>, tensor<768x768xi8>, tensor<768xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xi8>

  %444 = "mhlo_disc.dequantize"(%428, %150, %144) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xf32>
  %445 = "mhlo_disc.dequantize"(%430, %150, %144) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xf32>
  %446 = "mhlo_disc.dequantize"(%443, %150, %144) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xf32>

  %447 = "mhlo.add"(%444, %445) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
  %448 = "mhlo.add"(%447, %446) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>

  return %448: tensor<?x?x768xf32>
}

// CHECK-LABEL: func.func @quantized_dot_merging_dynamic_two
func.func @quantized_dot_merging_dynamic_two(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x768xf32> {  
  %10 = mhlo.constant dense<0> : tensor<768xi8>
  %130 = mhlo.constant dense_resource<__elided__> : tensor<768xi8>
  %131 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>
  %132 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>
  %133 = mhlo.constant dense_resource<__elided__> : tensor<768xi8>
  %134 = mhlo.constant dense_resource<__elided__> : tensor<768x768xi8>

  %143 = mhlo.constant dense<0.0741644949> : tensor<f32>
  %144 = mhlo.constant dense<0> : tensor<i32>
  %145 = mhlo.constant dense<0.0022647616> : tensor<f32>
  %146 = mhlo.constant dense<0.0447170027> : tensor<f32>
  %150 = mhlo.constant dense<0.0264002532> : tensor<f32>

  // CHECK: %[[C0:.*]] = "mhlo.concatenate"
  // CHECK: %[[T0:.*]] = "mhlo.broadcast_in_dim"
  // CHECK: %[[C1:.*]] = "mhlo.concatenate"
  // CHECK: %[[C2:.*]] = "mhlo.concatenate"
  // CHECK: %[[C3:.*]] = "mhlo.concatenate"
  // CHECK: %[[C4:.*]] = "mhlo.concatenate"
  // CHECK: %[[C5:.*]] = "mhlo.concatenate"
  // CHECK: %[[C6:.*]] = "mhlo.concatenate"
  // CHECK: %[[C7:.*]] = "mhlo.concatenate"
  // CHECK: %[[T2:.*]] = "mhlo_disc.custom_call_v2"
  // CHECK: %[[T3:.*]] = mhlo.real_dynamic_slice %[[T2]]
  // CHECK: %[[T4:.*]] = mhlo.real_dynamic_slice %[[T2]]
  // CHECK: %[[T6:.*]] = "mhlo_disc.dequantize"
  %427 = "mhlo_disc.quantize"(%arg0, %143, %144) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?xi8>
  %428 = "mhlo_disc.custom_call_v2"(%427, %134, %133, %143, %144, %145, %144, %146, %144) {call_target_name = "ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,d,s,d,s,d,s", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xi8>, tensor<768x768xi8>, tensor<768xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xi8>
  %430 = "mhlo_disc.custom_call_v2"(%427, %132, %10, %143, %144, %145, %144, %146, %144) {call_target_name = "ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,d,s,d,s,d,s", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xi8>, tensor<768x768xi8>, tensor<768xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xi8>

  %444 = "mhlo_disc.dequantize"(%428, %150, %144) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xf32>
  %445 = "mhlo_disc.dequantize"(%430, %150, %144) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x768xf32>

  %447 = "mhlo.add"(%444, %445) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>

  return %447: tensor<?x?x768xf32>
}