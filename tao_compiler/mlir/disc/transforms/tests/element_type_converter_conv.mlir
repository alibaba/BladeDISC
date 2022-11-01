// RUN: disc-opt -disc-element-type-converter=enable-fp16-conv=false %s | FileCheck %s --check-prefix=BASIC
// RUN: disc-opt -disc-element-type-converter=enable-fp16-conv=true %s | FileCheck %s --check-prefix=FP16


// CHECK-LABEL: @dynamic_conv_fp32

// BASIC: mhlo.dynamic_conv
// BASIC-NOT: f16

// FP16: mhlo.dynamic_conv
// FP16-SAME: f16
func.func @dynamic_conv_fp32(%arg0 : tensor<?x?x?x?xf32>, %arg1: tensor<64x3x7x7xf32>, %arg2: tensor<4xi32>) -> tensor<?x64x?x?xf32> {
  %1 = "mhlo.dynamic_conv"(%arg0, %arg1, %arg2) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>, disc.device = "gpu", feature_group_count = 1 : i64, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<?x?x?x?xf32>, tensor<64x3x7x7xf32>, tensor<4xi32>) -> tensor<?x64x?x?xf32>
  return %1 : tensor<?x64x?x?xf32>
}

// CHECK-LABEL: @conv_fp32

// BASIC: mhlo.convolution
// BASIC-NOT: f16

// FP16: mhlo.convolution
// FP16-SAME: f16
func.func @conv_fp32(%arg0 : tensor<?x?x?x?xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32> {
  %1 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>, disc.device = "gpu", feature_group_count = 1 : i64, padding = dense<[[3, 3], [3, 3]]> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<?x?x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
  return %1 : tensor<?x64x?x?xf32>
}
