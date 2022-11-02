// RUN: disc-opt --disc-lower-quantize-and-dequantize -split-input-file %s | FileCheck %s

// CHECK-LABEL: @quantize_signed_per_tensor
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xf32>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<f32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<i32>
func.func @quantize_signed_per_tensor(
    %input: tensor<?x32x32x6xf32>,
    %input_scale : tensor<f32>,
    %input_zero_point : tensor<i32>) -> tensor<?x32x32x6xi8> {

  // CHECK-DAG: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant  dense<1.270000e+02> : tensor<f32>

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[INPUT_SCALE]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[CASTED_ZERO_POINT:.*]] = mhlo.convert %[[INPUT_ZERO_POINT]] : (tensor<i32>) -> tensor<f32>
  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[CASTED_ZERO_POINT]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MIN:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MIN]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MAX:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MAX]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[T0:.*]] = mhlo.divide %[[INPUT]], %[[BCAST_SCALE]]
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[BCAST_ZERO_POINT]]
  // CHECK: %[[T2:.*]] = mhlo.round_nearest_even %[[T1]]
  // CHECK: %[[T3:.*]] = mhlo.clamp %[[T2]], %[[BCAST_MIN]], %[[BCAST_MAX]]
  // CHECK: %[[OUT:.*]] = mhlo.convert %[[T3]] : (tensor<?x32x32x6xf32>) -> tensor<?x32x32x6xi8>
  // CHECK: return %[[OUT]]

  %quantized_input = "mhlo_disc.quantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false,
      round_mode = 1
  } : (tensor<?x32x32x6xf32>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xi8>
  return %quantized_input : tensor<?x32x32x6xi8>
}

// -----

// CHECK-LABEL: @quantize_unsigned_per_tensor
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xf32>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<f32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<i32>
func.func @quantize_unsigned_per_tensor(
    %input: tensor<?x32x32x6xf32>,
    %input_scale : tensor<f32>,
    %input_zero_point : tensor<i32>) -> tensor<?x32x32x6xui8> {

  // CHECK-DAG: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant  dense<2.550000e+02> : tensor<f32>

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[INPUT_SCALE]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[CASTED_ZERO_POINT:.*]] = mhlo.convert %[[INPUT_ZERO_POINT]] : (tensor<i32>) -> tensor<f32>
  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[CASTED_ZERO_POINT]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MIN:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MIN]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MAX:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MAX]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[T0:.*]] = mhlo.divide %[[INPUT]], %[[BCAST_SCALE]]
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[BCAST_ZERO_POINT]]
  // CHECK: %[[T2:.*]] = mhlo.round_nearest_afz %[[T1]]
  // CHECK: %[[T3:.*]] = mhlo.clamp %[[T2]], %[[BCAST_MIN]], %[[BCAST_MAX]]
  // CHECK: %[[OUT:.*]] = mhlo.convert %[[T3]] : (tensor<?x32x32x6xf32>) -> tensor<?x32x32x6xui8>
  // CHECK: return %[[OUT]]

  %quantized_input = "mhlo_disc.quantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      quant_min = 0,
      quant_max = 255,
      use_dynamic = false
  } : (tensor<?x32x32x6xf32>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xui8>
  return %quantized_input : tensor<?x32x32x6xui8>
}

// -----

// CHECK-LABEL: @quantize_signed_per_channel
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xf32>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<?xf32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<?xi32>
func.func @quantize_signed_per_channel(
    %input: tensor<?x32x32x6xf32>,
    %input_scale : tensor<?xf32>,
    %input_zero_point : tensor<?xi32>) -> tensor<?x32x32x6xi8> {

  // CHECK-DAG: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant  dense<1.270000e+02> : tensor<f32>

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[INPUT_SCALE]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[CASTED_ZERO_POINT:.*]] = mhlo.convert %[[INPUT_ZERO_POINT]] : (tensor<?xi32>) -> tensor<?xf32>
  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[CASTED_ZERO_POINT]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[BCAST_MIN:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MIN]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MAX:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MAX]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[T0:.*]] = mhlo.divide %[[INPUT]], %[[BCAST_SCALE]]
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[BCAST_ZERO_POINT]]
  // CHECK: %[[T2:.*]] = mhlo.round_nearest_afz %[[T1]]
  // CHECK: %[[T3:.*]] = mhlo.clamp %[[T2]], %[[BCAST_MIN]], %[[BCAST_MAX]]
  // CHECK: %[[OUT:.*]] = mhlo.convert %[[T3]] : (tensor<?x32x32x6xf32>) -> tensor<?x32x32x6xi8>
  // CHECK: return %[[OUT]]

  %quantized_input = "mhlo_disc.quantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      quant_min = -128,
      quant_max = 127,
      use_dynamic = false
  } : (tensor<?x32x32x6xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x32x32x6xi8>
  return %quantized_input : tensor<?x32x32x6xi8>
}

// -----

// CHECK-LABEL: @quantize_unsigned_per_channel
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xf32>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<?xf32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<?xi32>
func.func @quantize_unsigned_per_channel(
    %input: tensor<?x32x32x6xf32>,
    %input_scale : tensor<?xf32>,
    %input_zero_point : tensor<?xi32>) -> tensor<?x32x32x6xui8> {

  // CHECK-DAG: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant  dense<2.550000e+02> : tensor<f32>

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[INPUT_SCALE]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[CASTED_ZERO_POINT:.*]] = mhlo.convert %[[INPUT_ZERO_POINT]] : (tensor<?xi32>) -> tensor<?xf32>
  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[CASTED_ZERO_POINT]], %[[INPUT_SHAPE]])
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[BCAST_MIN:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MIN]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_MAX:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[QUANT_MAX]], %[[INPUT_SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[T0:.*]] = mhlo.divide %[[INPUT]], %[[BCAST_SCALE]]
  // CHECK: %[[T1:.*]] = mhlo.add %[[T0]], %[[BCAST_ZERO_POINT]]
  // CHECK: %[[T2:.*]] = mhlo.round_nearest_even %[[T1]]
  // CHECK: %[[T3:.*]] = mhlo.clamp %[[T2]], %[[BCAST_MIN]], %[[BCAST_MAX]]
  // CHECK: %[[OUT:.*]] = mhlo.convert %[[T3]] : (tensor<?x32x32x6xf32>) -> tensor<?x32x32x6xui8>
  // CHECK: return %[[OUT]]

  %quantized_input = "mhlo_disc.quantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      quant_min = 0,
      quant_max = 255,
      use_dynamic = false,
      round_mode = 1
  } : (tensor<?x32x32x6xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x32x32x6xui8>
  return %quantized_input : tensor<?x32x32x6xui8>
}

// -----

// CHECK-LABEL: @dequantize_signed_per_tensor
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xi8>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<f32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<i32>
func.func @dequantize_signed_per_tensor(
    %input: tensor<?x32x32x6xi8>,
    %input_scale : tensor<f32>,
    %input_zero_point : tensor<i32>) -> tensor<?x32x32x6xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_SCALE]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_ZERO_POINT]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[CASTED_INPUT:.*]] = mhlo.convert %[[INPUT]] : (tensor<?x32x32x6xi8>) -> tensor<?x32x32x6xi32>
  // CHECK: %[[T0:.*]] = mhlo.subtract %[[CASTED_INPUT]], %[[BCAST_ZERO_POINT]] : tensor<?x32x32x6xi32>
  // CHECK: %[[T1:.*]] = mhlo.convert %[[T0]] : (tensor<?x32x32x6xi32>) -> tensor<?x32x32x6xf32>
  // CHECK: %[[T2:.*]] = mhlo.multiply %[[T1]], %[[BCAST_SCALE]] : tensor<?x32x32x6xf32>
  // CHECK: return %[[T2]]
  %quantized_input = "mhlo_disc.dequantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false
  } : (tensor<?x32x32x6xi8>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xf32>
  return %quantized_input : tensor<?x32x32x6xf32>
}

// -----

// CHECK-LABEL: @dequantize_unsigned_per_tensor
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xui8>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<f32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<i32>
func.func @dequantize_unsigned_per_tensor(
    %input: tensor<?x32x32x6xui8>,
    %input_scale : tensor<f32>,
    %input_zero_point : tensor<i32>) -> tensor<?x32x32x6xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_SCALE]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_ZERO_POINT]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<> : tensor<0xi64>

  // CHECK: %[[CASTED_INPUT:.*]] = mhlo.convert %[[INPUT]] : (tensor<?x32x32x6xui8>) -> tensor<?x32x32x6xi32>
  // CHECK: %[[T0:.*]] = mhlo.subtract %[[CASTED_INPUT]], %[[BCAST_ZERO_POINT]] : tensor<?x32x32x6xi32>
  // CHECK: %[[T1:.*]] = mhlo.convert %[[T0]] : (tensor<?x32x32x6xi32>) -> tensor<?x32x32x6xf32>
  // CHECK: %[[T2:.*]] = mhlo.multiply %[[T1]], %[[BCAST_SCALE]] : tensor<?x32x32x6xf32>
  // CHECK: return %[[T2]]
  %quantized_input = "mhlo_disc.dequantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false
  } : (tensor<?x32x32x6xui8>, tensor<f32>, tensor<i32>) -> tensor<?x32x32x6xf32>
  return %quantized_input : tensor<?x32x32x6xf32>
}

// -----

// CHECK-LABEL: @dequantize_signed_per_channel
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xi8>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<?xf32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<?xi32>
func.func @dequantize_signed_per_channel(
    %input: tensor<?x32x32x6xi8>,
    %input_scale : tensor<?xf32>,
    %input_zero_point : tensor<?xi32>) -> tensor<?x32x32x6xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_SCALE]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_ZERO_POINT]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[CASTED_INPUT:.*]] = mhlo.convert %[[INPUT]] : (tensor<?x32x32x6xi8>) -> tensor<?x32x32x6xi32>
  // CHECK: %[[T0:.*]] = mhlo.subtract %[[CASTED_INPUT]], %[[BCAST_ZERO_POINT]] : tensor<?x32x32x6xi32>
  // CHECK: %[[T1:.*]] = mhlo.convert %[[T0]] : (tensor<?x32x32x6xi32>) -> tensor<?x32x32x6xf32>
  // CHECK: %[[T2:.*]] = mhlo.multiply %[[T1]], %[[BCAST_SCALE]] : tensor<?x32x32x6xf32>
  // CHECK: return %[[T2]]
  %quantized_input = "mhlo_disc.dequantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      use_dynamic = false
  } : (tensor<?x32x32x6xi8>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x32x32x6xf32>
  return %quantized_input : tensor<?x32x32x6xf32>
}

// -----

// CHECK-LABEL: @dequantize_unsigned_per_channel
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x32x32x6xui8>
// CHECK-SAME: %[[INPUT_SCALE:.*]]: tensor<?xf32>
// CHECK-SAME: %[[INPUT_ZERO_POINT:.*]]: tensor<?xi32>
func.func @dequantize_unsigned_per_channel(
    %input: tensor<?x32x32x6xui8>,
    %input_scale : tensor<?xf32>,
    %input_zero_point : tensor<?xi32>) -> tensor<?x32x32x6xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT]]

  // CHECK: %[[BCAST_SCALE:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_SCALE]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[BCAST_ZERO_POINT:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[INPUT_ZERO_POINT]], %[[SHAPE]]
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>

  // CHECK: %[[CASTED_INPUT:.*]] = mhlo.convert %[[INPUT]] : (tensor<?x32x32x6xui8>) -> tensor<?x32x32x6xi32>
  // CHECK: %[[T0:.*]] = mhlo.subtract %[[CASTED_INPUT]], %[[BCAST_ZERO_POINT]] : tensor<?x32x32x6xi32>
  // CHECK: %[[T1:.*]] = mhlo.convert %[[T0]] : (tensor<?x32x32x6xi32>) -> tensor<?x32x32x6xf32>
  // CHECK: %[[T2:.*]] = mhlo.multiply %[[T1]], %[[BCAST_SCALE]] : tensor<?x32x32x6xf32>
  // CHECK: return %[[T2]]
  %quantized_input = "mhlo_disc.dequantize"(%input, %input_scale, %input_zero_point) {
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      use_dynamic = false
  } : (tensor<?x32x32x6xui8>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x32x32x6xf32>
  return %quantized_input : tensor<?x32x32x6xf32>
}
