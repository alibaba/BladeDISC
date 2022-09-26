// RUN: disc-opt --disc-convert-fake-quant-op -split-input-file %s | FileCheck %s

// CHECK-LABEL: @fake_quant_dynamic_conv
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x32x32x6xf32>, %[[WEIGHT:.*]]: tensor<3x3x3x16xf32>
func.func @fake_quant_dynamic_conv(
    %input: tensor<?x32x32x6xf32>, %weight: tensor<3x3x3x16xf32>, %padding: tensor<4xi32>,
    %input_scale : tensor<f32>, %input_zero_point : tensor<i32>,
    %weight_scale : tensor<?xf32>, %weight_zero_point : tensor<?xi32>,
    %result_scale : tensor<f32>, %result_zero_point : tensor<i32>
) -> tensor<?x8x6x16xf32> {
  // CHECK: %[[FAKE_QUANT_INPUT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[INPUT]]
  %fake_quant_input = "mhlo_disc.fake_quant"(%input, %input_scale, %input_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x32x32x6xf32>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xf32>

  // CHECK: %[[FAKE_QUANT_WEIGHT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[WEIGHT]]
  %fake_quant_weight = "mhlo_disc.fake_quant"(%weight, %weight_scale, %weight_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[2]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<3x3x3x16xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<3x3x3x16xf32>

  // CHECK: %[[RESULT:.*]] = "mhlo_disc.quantized_dynamic_conv"
  // CHECK-SAME: %[[FAKE_QUANT_INPUT]]
  // CHECK-SAME: %[[FAKE_QUANT_WEIGHT]]
  %result = "mhlo.dynamic_conv"(%fake_quant_input, %fake_quant_weight, %padding) {
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0, output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    batch_group_count = 1 : i64,
    feature_group_count = 2 : i64,
    padding = dense<[[1, 2], [3, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 3]> : tensor<2xi64>,
    window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<?x8x6x16xf32>

  // CHECK: %[[FAKE_QUANT_RESULT:.*]] = "mhlo_disc.dequantize"
  // CHECK-SAME: %[[RESULT]]
  %fake_quant_result = "mhlo_disc.fake_quant"(%result, %result_scale, %result_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x8x6x16xf32>, tensor<f32>, tensor<i32>) -> tensor<?x8x6x16xf32>

  // CHECK: return %[[FAKE_QUANT_RESULT]] : tensor<?x8x6x16xf32>
  return %fake_quant_result : tensor<?x8x6x16xf32>
}

// -----

// CHECK-LABEL: @fake_quant_conv
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x32x32x6xf32>, %[[WEIGHT:.*]]: tensor<3x3x3x16xf32>
func.func @fake_quant_conv(
    %input: tensor<?x32x32x6xf32>, %weight: tensor<3x3x3x16xf32>,
    %input_scale : tensor<f32>, %input_zero_point : tensor<i32>,
    %weight_scale : tensor<?xf32>, %weight_zero_point : tensor<?xi32>,
    %result_scale : tensor<f32>, %result_zero_point : tensor<i32>
) -> tensor<?x8x6x16xf32> {
  // CHECK: %[[FAKE_QUANT_INPUT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[INPUT]]
  %fake_quant_input = "mhlo_disc.fake_quant"(%input, %input_scale, %input_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x32x32x6xf32>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xf32>

  // CHECK: %[[FAKE_QUANT_WEIGHT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[WEIGHT]]
  %fake_quant_weight = "mhlo_disc.fake_quant"(%weight, %weight_scale, %weight_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[2]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<3x3x3x16xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<3x3x3x16xf32>

  // CHECK: %[[RESULT:.*]] = "mhlo_disc.quantized_dynamic_conv"
  // CHECK-SAME: %[[FAKE_QUANT_INPUT]]
  // CHECK-SAME: %[[FAKE_QUANT_WEIGHT]]
  %result = "mhlo.convolution"(%fake_quant_input, %fake_quant_weight) {
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0, output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    batch_group_count = 1 : i64,
    feature_group_count = 2 : i64,
    padding = dense<[[1, 2], [3, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 3]> : tensor<2xi64>,
    window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<?x8x6x16xf32>

  // CHECK: %[[FAKE_QUANT_RESULT:.*]] = "mhlo_disc.dequantize"
  // CHECK-SAME: %[[RESULT]]
  %fake_quant_result = "mhlo_disc.fake_quant"(%result, %result_scale, %result_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x8x6x16xf32>, tensor<f32>, tensor<i32>) -> tensor<?x8x6x16xf32>

  // CHECK: return %[[FAKE_QUANT_RESULT]] : tensor<?x8x6x16xf32>
  return %fake_quant_result : tensor<?x8x6x16xf32>
}

// -----

// CHECK-LABEL: @fake_quant_dot
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?xf32>, %[[WEIGHT:.*]]: tensor<?x?xf32>
func.func @fake_quant_dot(
  %input: tensor<?x?xf32>, %weight: tensor<?x?xf32>,
  %input_scale : tensor<f32>, %input_zero_point : tensor<i32>,
  %weight_scale : tensor<?xf32>, %weight_zero_point : tensor<?xi32>,
  %result_scale : tensor<f32>, %result_zero_point : tensor<i32>
) -> tensor<?x?xf32> {
  // CHECK: %[[FAKE_QUANT_INPUT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[INPUT]]
  %fake_quant_input = "mhlo_disc.fake_quant"(%input, %input_scale, %input_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: %[[FAKE_QUANT_WEIGHT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[WEIGHT]]
  %fake_quant_weight = "mhlo_disc.fake_quant"(%weight, %weight_scale, %weight_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>

  // CHECK: %[[RESULT:.*]] = "mhlo_disc.quantized_dot_general"
  // CHECK-SAME: %[[FAKE_QUANT_INPUT]]
  // CHECK-SAME: %[[FAKE_QUANT_WEIGHT]]
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  %result = "mhlo.dot"(%fake_quant_input, %fake_quant_weight)
   : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: %[[FAKE_QUANT_RESULT:.*]] = "mhlo_disc.dequantize"
  // CHECK-SAME: %[[RESULT]]
  %fake_quant_result = "mhlo_disc.fake_quant"(%result, %result_scale, %result_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: return %[[FAKE_QUANT_RESULT]]
  return %fake_quant_result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @fake_quant_dot_general
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?x?x?xf32>, %[[WEIGHT:.*]]: tensor<?x?x?x?xf32>
func.func @fake_quant_dot_general(
  %input: tensor<?x?x?x?xf32>, %weight: tensor<?x?x?x?xf32>,
  %input_scale : tensor<f32>, %input_zero_point : tensor<i32>,
  %weight_scale : tensor<?xf32>, %weight_zero_point : tensor<?xi32>,
  %result_scale : tensor<f32>, %result_zero_point : tensor<i32>
) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[FAKE_QUANT_INPUT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[INPUT]]
  %fake_quant_input = "mhlo_disc.fake_quant"(%input, %input_scale, %input_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>

  // CHECK: %[[FAKE_QUANT_WEIGHT:.*]] = "mhlo_disc.quantize"
  // CHECK-SAME: %[[WEIGHT]]
  %fake_quant_weight = "mhlo_disc.fake_quant"(%weight, %weight_scale, %weight_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>

  // CHECK: %[[RESULT:.*]] = "mhlo_disc.quantized_dot_general"
  // CHECK-SAME: %[[FAKE_QUANT_INPUT]]
  // CHECK-SAME: %[[FAKE_QUANT_WEIGHT]]
  // CHECK-SAME: lhs_batching_dimensions = [0, 1]
  // CHECK-SAME: rhs_batching_dimensions = [0, 1]
  // CHECK-SAME: lhs_contracting_dimensions = [3]
  // CHECK-SAME: rhs_contracting_dimensions = [3]
  %result = "mhlo.dot_general"(%fake_quant_input, %fake_quant_weight) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >}
   : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: %[[FAKE_QUANT_RESULT:.*]] = "mhlo_disc.dequantize"
  // CHECK-SAME: %[[RESULT]]
  %fake_quant_result = "mhlo_disc.fake_quant"(%result, %result_scale, %result_zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>

  // CHECK: return %[[FAKE_QUANT_RESULT]]
  return %fake_quant_result : tensor<?x?x?x?xf32>
}

