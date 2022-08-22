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
      use_dynamic = false
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
      use_dynamic = false
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
      use_dynamic = false
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
      use_dynamic = false
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @dequantize_per_tensor
func.func @dequantize_per_tensor(%input : tensor<?x?x?x?xi8>, %scale : tensor<f32>, %zero_point : tensor<i32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false
  } : (tensor<?x?x?x?xi8>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @dequantize_per_channel
func.func @dequantize_per_channel(%input : tensor<?x?x?x?xi8>, %scale : tensor<?xf32>, %zero_point : tensor<?xi32>) -> tensor<?x?x?x?xf32> {
  %out = "mhlo_disc.dequantize"(%input, %scale, %zero_point) {
      use_symmetric = true,
      axis = dense<[1]> : tensor<1xi64>,
      use_dynamic = false
  } : (tensor<?x?x?x?xi8>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?x?x?xf32>
  return %out : tensor<?x?x?x?xf32>
}
