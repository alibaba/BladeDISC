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