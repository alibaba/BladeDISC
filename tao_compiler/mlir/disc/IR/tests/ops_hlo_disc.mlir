// RUN: disc-opt -split-input-file %s | FileCheck %s

// CHECK-LABEL: @fake_quant_per_tensor
func.func @fake_quant_per_tensor(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.fake_quant"(%input, %scale, %zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      num_bits = 8,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false,
      round_mode = 1
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @fake_quant_per_channel
func.func @fake_quant_per_channel(%input : tensor<?x?x?x?xf32>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.fake_quant"(%input, %scale, %zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false,
      round_mode = 0
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @quantize_per_tensor
func.func @quantize_per_tensor(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xi8> {
  %out = "mhlo_disc.quantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false,
      round_mode = 1
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @quantize_per_channel
func.func @quantize_per_channel(%input : tensor<?x?x?x?xf32>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xi8> {
  %out = "mhlo_disc.quantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false,
      round_mode = 0
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @dequantize_per_tensor
func.func @dequantize_per_tensor(%input : tensor<?x?x?x?xi8>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      round_mode = 1
  } : (tensor<?x?x?x?xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @dequantize_per_channel
func.func @dequantize_per_channel(%input : tensor<?x?x?x?xi8>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      use_dynamic = false,
      round_mode = 0
  } : (tensor<?x?x?x?xi8>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @per_tensor_quantized_dot_general
func.func @per_tensor_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?x?xi8>,
                                            %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                            %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                            %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?xi8> {
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<?x?xi8>, tensor<?x?xi8>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @per_channel_quantized_dot_general
func.func @per_channel_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?x?xi8>,
                                             %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                             %weight_scale: tensor<?xf32>, %weight_zero_point: tensor<?xi32>,
                                             %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?xi8> {
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<?x?xi8>, tensor<?x?xi8>,
       tensor<f32>, tensor<i32>,
       tensor<?xf32>, tensor<?xi32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @per_channel_quantized_dynamic_conv
func.func @per_channel_quantized_dynamic_conv(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xi32>,
                                             %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                             %weight_scale: tensor<?xf32>, %weight_zero_point: tensor<?xi32>,
                                             %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  %out = "mhlo_disc.quantized_dynamic_conv"(%input, %weight, %padding,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      use_dynamic = false,
      dimension_numbers = #mhlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 1,
        input_spatial_dimensions = [2, 3],
        kernel_input_feature_dimension = 1,
        kernel_output_feature_dimension = 0,
        kernel_spatial_dimensions = [2, 3],
        output_batch_dimension = 0,
        output_feature_dimension = 1,
        output_spatial_dimensions = [2, 3]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<3> : tensor<2xi64>
  } : (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, tensor<4xi32>,
       tensor<f32>, tensor<i32>,
       tensor<?xf32>, tensor<?xi32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @per_tensor_quantized_dynamic_conv
func.func @per_tensor_quantized_dynamic_conv(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xi32>,
                                             %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                             %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                             %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  %out = "mhlo_disc.quantized_dynamic_conv"(%input, %weight, %padding,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dimension_numbers = #mhlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 1,
        input_spatial_dimensions = [2, 3],
        kernel_input_feature_dimension = 1,
        kernel_output_feature_dimension = 0,
        kernel_spatial_dimensions = [2, 3],
        output_batch_dimension = 0,
        output_feature_dimension = 1,
        output_spatial_dimensions = [2, 3]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<3> : tensor<2xi64>
  } : (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, tensor<4xi32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: func.func @sparse_reshape_basic
func.func @sparse_reshape_basic(%input_indices : tensor<?x?xi64>, %input_shape: tensor<?xi64>, %new_shape: tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>) {
  %output_indices, %output_shape = "mhlo_disc.sparse_reshape"(%input_indices, %input_shape, %new_shape) {} : (tensor<?x?xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>)
  return %output_indices, %output_shape: tensor<?x?xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func.func @sparse_fill_empty_rows_basic
func.func @sparse_fill_empty_rows_basic(%indices: tensor<?x?xi64>, %values: tensor<?xi64>, %dense_shape: tensor<2xi64>, %default_value: tensor<i64>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>) {
  %output_indices, %output_values, %empty_row_indicator, %reverse_index_map, %output_elements = "mhlo_disc.sparse_fill_empty_rows"(%indices, %values, %dense_shape, %default_value) {} : (tensor<?x?xi64>, tensor<?xi64>, tensor<2xi64>, tensor<i64>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>)
  return %output_indices, %output_values, %empty_row_indicator, %reverse_index_map, %output_elements: tensor<?x?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func.func @sparse_segment_mean_basic
func.func @sparse_segment_mean_basic(%data: tensor<?x?xf32>, %indices: tensor<?xi32>, %segment_ids: tensor<?xi32>) -> (tensor<?x?xf32>) {
  %output = "mhlo_disc.sparse_segment_reduction"(%data, %indices, %segment_ids) { reduction_mode = 1 } : (tensor<?x?xf32>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?x?xf32>)
  return %output : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @where
func.func @where(%input: tensor<?x?xf32>) -> (tensor<?x?xi64>, tensor<1xi64>) {
  %index, %num_output_elements = "mhlo_disc.where"(%input) {} : (tensor<?x?xf32>) -> (tensor<?x?xi64>, tensor<1xi64>)
  return %index, %num_output_elements: tensor<?x?xi64>, tensor<1xi64>
}

// -----

// CHECK-LABEL: func.func @custom_call_v2
func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = "Ab",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}
