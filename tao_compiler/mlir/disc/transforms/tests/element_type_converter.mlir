// RUN: disc-opt -disc-element-type-converter=enable-fp16-gemm=false %s | FileCheck %s --check-prefix=BASIC
// RUN: disc-opt -disc-element-type-converter=enable-fp16-gemm=true %s | FileCheck %s --check-prefix=FP16

// CHECK-LABEL: @dot_fp32

// Test with `enable_fp16_gemm=false`
// BASIC: mhlo.dot_general
// BASIC-NOT: f16

// Test with `enable_fp16_gemm=true`
// FP16: mhlo.dot_general
// FP16-SAME: f16
func.func @dot_fp32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_batching_dimensions = [], rhs_contracting_dimensions = [1]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @rank2_colunm_reduction_i1
func.func @rank2_colunm_reduction_i1(%arg0: tensor<?x?xi1>) -> tensor<?xi1> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<i1>, %arg2: tensor<i1>):
    %2 = mhlo.add %arg1, %arg2 : tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?x?xi1>, tensor<i1>) -> tensor<?xi1>
  // BASIC: mhlo.convert({{.*}}) : (tensor<?x?xi1>) -> tensor<?x?xi32>
  // BASIC-NEXT: mhlo.reduce
  // BASIC-NOT: i1
  // BASIC: mhlo.convert({{.*}}) : (tensor<?xi32>) -> tensor<?xi1>
  return %1 : tensor<?xi1>
}

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
