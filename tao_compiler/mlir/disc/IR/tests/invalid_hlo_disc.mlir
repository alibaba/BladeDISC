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
