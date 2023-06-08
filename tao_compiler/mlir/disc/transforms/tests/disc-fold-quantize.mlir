// RUN: disc-opt --canonicalize -split-input-file %s | FileCheck %s

// int8 normal
func.func @quantize_per_channel_symmetric_int8() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10, 20, 30], [20, 25, 30]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<0> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_tensor_symmetric_int8() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.1> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_channel_asymmetric_uint8() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[20, 30, 40], [40, 45, 50]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}


func.func @quantize_per_tensor_asymmetric_uint8() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[20, 30, 40], [50, 60, 70]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.1> : tensor<f32>
  %zero_point =  mhlo.constant dense<10> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}

func.func @quantize_per_channel_symmetric_int8_fp16() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10, 20, 30], [20, 25, 30]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>
  %scale = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf16>
  %zero_point =  mhlo.constant dense<0> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf16>, tensor<2xf16>, tensor<2xi32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_tensor_symmetric_int8_fp16() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>
  %scale = mhlo.constant dense<0.1> : tensor<f16>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf16>, tensor<f16>, tensor<i32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_channel_asymmetric_uint8_fp16() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[20, 30, 40], [40, 45, 50]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>
  %scale = mhlo.constant dense<[0.1, 0.2]> : tensor<2xf16>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf16>, tensor<2xf16>, tensor<2xi32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}


func.func @quantize_per_tensor_asymmetric_uint8_fp16() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[20, 30, 40], [50, 60, 70]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>
  %scale = mhlo.constant dense<0.1> : tensor<f16>
  %zero_point =  mhlo.constant dense<10> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf16>, tensor<f16>, tensor<i32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}


// int8 outside quant range
func.func @quantize_per_channel_symmetric_int8_outside_quant_range() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[100, 127, 127], [-100, -125, -128]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, -0.04]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<0> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_tensor_symmetric_int8_outside_quant_range() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[100, 127, 127], [127, 127, 127]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.01> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}


func.func @quantize_per_channel_asymmetric_uint8_outside_quant_range() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[110, 210, 255], [0, 0, 0]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, -0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}


func.func @quantize_per_tensor_asymmetric_uint8_outside_quant_range() ->  tensor<2x3xui8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[110, 210, 255], [255, 255, 255]]> : tensor<2x3xui8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.01> : tensor<f32>
  %zero_point =  mhlo.constant dense<10> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xui8>
  return %y : tensor<2x3xui8>
}


// int16 normal
func.func @quantize_per_channel_symmetric_int16() ->  tensor<2x3xi16> {
  // CHECK{LITERAL}: mhlo.constant dense<[[100, 200, 300], [200, 250, 300]]> : tensor<2x3xi16>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, 0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<0> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 32767 : i64, quant_min = -32768 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xi16>
  return %y : tensor<2x3xi16>
}


func.func @quantize_per_tensor_symmetric_int16() ->  tensor<2x3xi16> {
  // CHECK{LITERAL}: mhlo.constant dense<[[100, 200, 300], [400, 500, 600]]> : tensor<2x3xi16>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.01> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 32767 : i64, quant_min = -32768 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi16>
  return %y : tensor<2x3xi16>
}


func.func @quantize_per_channel_asymmetric_uint16() ->  tensor<2x3xui16> {
  // CHECK{LITERAL}: mhlo.constant dense<[[110, 210, 310], [220, 270, 320]]> : tensor<2x3xui16>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, 0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 65535 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui16>
  return %y : tensor<2x3xui16>
}


func.func @quantize_per_tensor_asymmetric_uint16() ->  tensor<2x3xui16> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10010, 20010, 30010], [40010, 50010, 60010]]> : tensor<2x3xui16>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.0001> : tensor<f32>
  %zero_point =  mhlo.constant dense<10> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 65535 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xui16>
  return %y : tensor<2x3xui16>
}


// int16 outside quantization range
func.func @quantize_per_channel_asymmetric_uint16_outside_quant_range() ->  tensor<2x3xui16> {
  // CHECK{LITERAL}: mhlo.constant dense<[[65535, 65535, 65535], [0, 0, 0]]> : tensor<2x3xui16>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.00001, -0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 65535 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui16>
  return %y : tensor<2x3xui16>
}


// int32 normal
func.func @quantize_per_channel_symmetric_int32() ->  tensor<2x3xi32> {
  // CHECK{LITERAL}: mhlo.constant dense<[[1000000, 2000000, 3000000], [2000000, 2500000, 3000000]]> : tensor<2x3xi32>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[10000.0, 20000.0, 30000.0], [40000.0, 50000.0, 60000.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, 0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<0> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 2147483647 : i64, quant_min = -2147483648 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %y : tensor<2x3xi32>
}


func.func @quantize_per_tensor_symmetric_int32() ->  tensor<2x3xi32> {
  // CHECK{LITERAL}: mhlo.constant dense<[[1000000, 2000000, 3000000], [4000000, 5000000, 6000000]]> : tensor<2x3xi32>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[10000.0, 20000.0, 30000.0], [40000.0, 50000.0, 60000.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.01> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 2147483647 : i64, quant_min = -2147483648 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi32>
  return %y : tensor<2x3xi32>
}


func.func @quantize_per_channel_asymmetric_uint32() ->  tensor<2x3xui32> {
  // CHECK{LITERAL}: mhlo.constant dense<[[1000010, 2000010, 3000010], [2000020, 2500020, 3000020]]> : tensor<2x3xui32>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[10000.0, 20000.0, 30000.0], [40000.0, 50000.0, 60000.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.01, 0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 4294967295 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui32>
  return %y : tensor<2x3xui32>
}


func.func @quantize_per_tensor_asymmetric_uint32() ->  tensor<2x3xui32> {
  // CHECK{LITERAL}: mhlo.constant dense<[[1000010, 2000010, 3000010], [4000010, 5000010, 6000010]]> : tensor<2x3xui32>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[10000.0, 20000.0, 30000.0], [40000.0, 50000.0, 60000.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<0.01> : tensor<f32>
  %zero_point =  mhlo.constant dense<10> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 4294967295 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xui32>
  return %y : tensor<2x3xui32>
}


// int32 outside quant range
func.func @quantize_per_channel_asymmetric_uint32_outside_quant_range() ->  tensor<2x3xui32> {
  // CHECK{LITERAL}: mhlo.constant dense<[[4294967295, 200010, 300010], [0, 0, 0]]> : tensor<2x3xui32>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1000000.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.00001, -0.02]> : tensor<2xf32>
  %zero_point =  mhlo.constant dense<[10, 20]> : tensor<2xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<0> : tensor<1xi64>, quant_max = 4294967295 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x3xui32>
  return %y : tensor<2x3xui32>
}


// test round mode
func.func @quantize_per_tensor_symmetric_int8_round_even() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[2, 2, 4], [-2, -2, -4]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.5, 2.5, 3.5], [-1.5, -2.5, -3.5]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<1.0> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

func.func @quantize_per_tensor_symmetric_int8_round_away_from_zero() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[2, 3, 4], [-2, -3, -4]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.5, 2.5, 3.5], [-1.5, -2.5, -3.5]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<1.0> : tensor<f32>
  %zero_point =  mhlo.constant dense<0> : tensor<i32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 0 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// test different axis
func.func @quantize_per_channel_symmetric_int8_axis1() ->  tensor<2x3xi8> {
  // CHECK{LITERAL}: mhlo.constant dense<[[10, 10, -30], [40, 25, -60]]> : tensor<2x3xi8>
  // CHECK-NOT: mhlo_disc.quantize
  %x = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %scale = mhlo.constant dense<[0.1, 0.2, -0.1]> : tensor<3xf32>
  %zero_point =  mhlo.constant dense<0> : tensor<3xi32>
  %y = "mhlo_disc.quantize"(%x, %scale, %zero_point) {axis = dense<1> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xi32>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}