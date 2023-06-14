// RUN: disc-opt --disc-transform-weight-data-layout-for-weight-only-quant -split-input-file %s | FileCheck %s


func.func @convert_weight_layout_convert(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x128xf16> {
  // CHECK: weight_should_be_reordered = false
  %0 = mhlo.constant dense<0> : tensor<64x128xui8>
  %1 = mhlo.constant dense<127> : tensor<128xi32>
  %2 = mhlo.constant dense_resource<__elided__> : tensor<128xf16>
  %3 = "mhlo_disc.custom_call_v2"(%arg0, %0, %2, %1) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,ab,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,ba,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf16>, tensor<64x128xui8>, tensor<128xf16>, tensor<128xi32>) -> tensor<?x?x128xf16>
  return %3 : tensor<?x?x128xf16>
 }

func.func @test_wrong_call_target_name(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x128xf16> {
  // CHECK: weight_should_be_reordered = true
  %0 = mhlo.constant dense<0> : tensor<64x128xui8>
  %1 = mhlo.constant dense<127> : tensor<128xi32>
  %2 = mhlo.constant dense_resource<__elided__> : tensor<128xf16>
  %3 = "mhlo_disc.custom_call_v2"(%arg0, %0, %2, %1) {call_target_name = "ral_pdll_weight_only_qgemm11", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,ab,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,ba,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf16>, tensor<64x128xui8>, tensor<128xf16>, tensor<128xi32>) -> tensor<?x?x128xf16>
  return %3 : tensor<?x?x128xf16>
 }

func.func @test_wrong_weight_shape(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x128xf16> {
  // CHECK: weight_should_be_reordered = true
  %0 = mhlo.constant dense<0> : tensor<32x64x128xui8>
  %1 = mhlo.constant dense<127> : tensor<128xi32>
  %2 = mhlo.constant dense_resource<__elided__> : tensor<128xf16>
  %3 = "mhlo_disc.custom_call_v2"(%arg0, %0, %2, %1) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,abc,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,cba,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf16>, tensor<32x64x128xui8>, tensor<128xf16>, tensor<128xi32>) -> tensor<?x?x128xf16>
  return %3 : tensor<?x?x128xf16>
 }

func.func @test_wrong_weight_dtype(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x128xf16> {
  // CHECK: weight_should_be_reordered = true
  %0 = mhlo.constant dense<0> : tensor<32x64x128xi8>
  %1 = mhlo.constant dense<127> : tensor<128xi32>
  %2 = mhlo.constant dense_resource<__elided__> : tensor<128xf16>
  %3 = "mhlo_disc.custom_call_v2"(%arg0, %0, %2, %1) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,abc,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,cba,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf16>, tensor<32x64x128xi8>, tensor<128xf16>, tensor<128xi32>) -> tensor<?x?x128xf16>
  return %3 : tensor<?x?x128xf16>
 }
 