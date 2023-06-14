// RUN: disc-opt -split-input-file %s -verify-diagnostics

func.func @fake_quant_per_tensor(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.fake_quant"(%input, %scale, %zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @fake_quant_per_channel(%input : tensor<?x?x?x?xf32>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.fake_quant"(%input, %scale, %zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[1,2]> : tensor<2xi64>,
      num_bits = 8,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @fake_quant_mismatch_scale_and_zero_point(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{scale and zero_point have mismatch rank}}
  %out = "mhlo_disc.fake_quant"(%input, %scale, %zero_point) {
      use_signed = true,
      use_symmetric = true,
      axis = dense<[2]> : tensor<1xi64>,
      num_bits = 8,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @quantize_per_tensor(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.quantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

func.func @quantize_per_channel(%input : tensor<?x?x?x?xf32>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.quantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1,2]> : tensor<2xi64>,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

func.func @quantize_mismatch_scale_and_zero_point(%input : tensor<?x?x?x?xf32>, %scale : tensor<f32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{scale and zero_point have mismatch rank}}
  %out = "mhlo_disc.quantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[2]> : tensor<1xi64>,
      quant_min = -111,
      quant_max = 111,
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<?xi32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

func.func @dequantize_per_tensor(%input : tensor<?x?x?x?xi8>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      use_dynamic = false
  } : (tensor<?x?x?x?xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @dequantize_per_channel(%input : tensor<?x?x?x?xi8>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{num of quantized axes (len(axis)) is not equal to the rank of scale tensor}}
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1,2]> : tensor<2xi64>,
      use_dynamic = false
  } : (tensor<?x?x?x?xi8>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @dequantize_mismatch_scale_and_zero_point(%input : tensor<?x?x?x?xi8>, %scale : tensor<f32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1 {{scale and zero_point have mismatch rank}}
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[2]> : tensor<1xi64>,
      use_dynamic = false
  } : (tensor<?x?x?x?xi8>, tensor<f32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

func.func @mismatch_rank_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?xi8>,
                                 %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                 %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                 %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?xi8> {
  // expected-error@+1 {{input, weight and result should have the same rank}}
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<?x?xi8>, tensor<?xi8>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

func.func @quantize_per_channel_input_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?x?xi8>,
                                 %input_scale: tensor<?xf32>, %input_zero_point: tensor<?xi32>,
                                 %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                 %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?xi8> {
  // expected-error@+1 {{input_scale and input_zero_point only support per-tensor quantization}}
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<?x?xi8>, tensor<?x?xi8>,
       tensor<?xf32>, tensor<?xi32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

func.func @quantize_per_channel_result_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?x?xi8>,
                                 %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                 %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                 %result_scale: tensor<?xf32>, %result_zero_point: tensor<?xi32>) -> tensor<?x?xi8> {
  // expected-error@+1 {{result_scale and result_zero_point only support per-tensor quantization}}
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
       tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

func.func @mismatch_rank_per_tensor_quantized_dynamic_conv(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?xi8>, %padding : tensor<4xi32>,
                                                           %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                                           %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                                           %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{input, weight and result should have the same rank}}
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
  } : (tensor<?x?x?x?xi8>, tensor<?x?x?xi8>, tensor<4xi32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

func.func @quantize_per_channel_input_quantized_dynamic_conv(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xi32>,
                                                           %input_scale: tensor<?xf32>, %input_zero_point: tensor<?xi32>,
                                                           %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                                           %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{input_scale and input_zero_point only support per-tensor quantization}}
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
       tensor<?xf32>, tensor<?xi32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

func.func @quantize_per_channel_result_quantized_dynamic_conv(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xi32>,
                                                              %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                                              %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                                              %result_scale: tensor<?xf32>, %result_zero_point: tensor<?xi32>) -> tensor<?x?x?x?xi8> {
  // expected-error@+1 {{result_scale and result_zero_point only support per-tensor quantization}}
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
       tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: func.func @sparse_reshape_rank1_input
func.func @sparse_reshape_rank1_input(%input_indices : tensor<?xi64>, %input_shape: tensor<?xi64>, %new_shape: tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>) {
  // expected-error@+1 {{Input/Output indices must be a matrix.}}
  %output_indices, %output_shape = "mhlo_disc.sparse_reshape"(%input_indices, %input_shape, %new_shape) {} : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>)
  return %output_indices, %output_shape: tensor<?x?xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func.func @sparse_reshape_non_vector_shape
func.func @sparse_reshape_non_vector_shape(%input_indices : tensor<?x?xi64>, %input_shape: tensor<?x?xi64>, %new_shape: tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>) {
  // expected-error@+1 {{Input/Output shape and new shape must be a vector.}}
  %output_indices, %output_shape = "mhlo_disc.sparse_reshape"(%input_indices, %input_shape, %new_shape) {} : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>)
  return %output_indices, %output_shape: tensor<?x?xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func.func @sparse_reshape_non_matrix_indices
func.func @sparse_fill_empty_rows_basic(%indices: tensor<?xi64>, %values: tensor<?xi64>, %dense_shape: tensor<?xi64>, %default_value: tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>) {
  // expected-error@+1 {{indices must be a matrix}}
  %output_indices, %output_values, %empty_row_indicator, %reverse_index_map, %output_elements = "mhlo_disc.sparse_fill_empty_rows"(%indices, %values, %dense_shape, %default_value) {} : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>)
  return %output_indices, %output_values, %empty_row_indicator, %reverse_index_map, %output_elements: tensor<?xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func.func @sparse_segment_mean_matrix_indices
func.func @sparse_segment_mean_matrix_indices(%data: tensor<?x?xf32>, %indices: tensor<?x?xi32>, %segment_ids: tensor<?xi32>) -> (tensor<?x?xf32>) {
  // expected-error@+1 {{indices should be a vector}}
  %output = "mhlo_disc.sparse_segment_reduction"(%data, %indices, %segment_ids) { reduction_mode = 0 } : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<?xi32>) -> (tensor<?x?xf32>)
  return %output : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @sparse_segment_mean_matrix_segment_ids
func.func @sparse_segment_mean_matrix_segment_ids(%data: tensor<?x?xf32>, %indices: tensor<?xi32>, %segment_ids: tensor<?x?xi32>) -> (tensor<?x?xf32>) {
  // expected-error@+1 {{segment_ids should be a vector}}
  %output = "mhlo_disc.sparse_segment_reduction"(%data, %indices, %segment_ids) {} : (tensor<?x?xf32>, tensor<?xi32>, tensor<?x?xi32>) -> (tensor<?x?xf32>)
  return %output : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @sparse_segment_mean_no_match_indices_segment_ids
func.func @sparse_segment_mean_no_match_indices_segment_ids(%data: tensor<?x?xf32>, %indices: tensor<6xi32>, %segment_ids: tensor<4xi32>) -> (tensor<?x?xf32>) {
  // expected-error@+1 {{segment_ids and indices should have same size}}
  %output = "mhlo_disc.sparse_segment_reduction"(%data, %indices, %segment_ids) {} : (tensor<?x?xf32>, tensor<6xi32>, tensor<4xi32>) -> (tensor<?x?xf32>)
  return %output : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @where_invalid_output
func.func @where_invalid_output(%input: tensor<?x?xf32>) -> (tensor<?xi64>, tensor<1xi64>) {
  // expected-error@+1 {{output must be a matrix}}
  %index, %num_output_elements = "mhlo_disc.where"(%input) {} : (tensor<?x?xf32>) -> (tensor<?xi64>, tensor<1xi64>)
  return %index, %num_output_elements: tensor<?xi64>, tensor<1xi64>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch number of layouts for input_layouts and expected_input_layouts}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "",
      output_layouts = "AB",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch number of layouts for input_layouts and expected_input_layouts}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = "AB",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch number of input layouts and number of inputs}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB,AB",
      expected_input_layouts = "AB,AB,AB",
      output_layouts = "AB",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch number of layouts for output_layouts and expected_output_layouts}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = ",",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch number of output layouts and number of outputs}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = "AB,AB",
      expected_output_layouts = "AB,AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch input layout or expected input layout setting}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "ABC,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = "AB",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

func.func @custom_call_v2(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch output layout or expected output layout setting}}
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "AB,AB",
      output_layouts = "AB",
      expected_output_layouts = "A"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}
