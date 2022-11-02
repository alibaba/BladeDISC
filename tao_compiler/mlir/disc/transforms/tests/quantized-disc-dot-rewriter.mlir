// RUN: disc-opt %s -quantized-disc-dot-rewriter -split-input-file | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x128xf32> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  %2 = mhlo.constant dense<1.270000e+02> : tensor<f32>
  %3 = shape.const_shape [64, 128] : tensor<2xindex>
  %4 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %5 = mhlo.constant dense<0> : tensor<128xi32>
  %6 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
  %7 = mhlo.constant dense<1.000000e+00> : tensor<64x128xf32>
  %8 = mhlo.constant dense<2.000000e+00> : tensor<f32>
  %9 = mhlo.constant dense<0> : tensor<i32>
  %10 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %11 = "mhlo.dynamic_broadcast_in_dim"(%8, %10) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %12 = "mhlo.dynamic_broadcast_in_dim"(%0, %10) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %13 = "mhlo.dynamic_broadcast_in_dim"(%1, %10) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %14 = "mhlo.dynamic_broadcast_in_dim"(%2, %10) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %15 = mhlo.divide %arg0, %11 : tensor<?x?xf32>
  %16 = mhlo.add %15, %12 : tensor<?x?xf32>
  %17 = mhlo.round_nearest_afz %16 : tensor<?x?xf32>
  %18 = mhlo.clamp %17, %13, %14 : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %19 = mhlo.convert %18 : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %20 = "mhlo.dynamic_broadcast_in_dim"(%6, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>, tensor<2xindex>) -> tensor<64x128xf32>
  %21 = "mhlo.dynamic_broadcast_in_dim"(%4, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>, tensor<2xindex>) -> tensor<64x128xf32>
  %22 = "mhlo.dynamic_broadcast_in_dim"(%1, %3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<64x128xf32>
  %23 = "mhlo.dynamic_broadcast_in_dim"(%2, %3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<64x128xf32>
  %24 = mhlo.divide %7, %20 : tensor<64x128xf32>
  %25 = mhlo.add %24, %21 : tensor<64x128xf32>
  %26 = mhlo.round_nearest_afz %25 : tensor<64x128xf32>
  %27 = mhlo.clamp %26, %22, %23 : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %28 = mhlo.convert %27 : (tensor<64x128xf32>) -> tensor<64x128xi8>
  // CHECK: %[[T29:.*]] = "mhlo_disc.quantized_dot_general"
  // CHECK-SAME: rhs_contracting_dimensions = [1]
  %29 = "mhlo_disc.quantized_dot_general"(%19, %28, %8, %9, %6, %5, %8, %9) {axis = dense<1> : tensor<1xi64>, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, use_dynamic = false, use_symmetric = true} : (tensor<?x?xi8>, tensor<64x128xi8>, tensor<f32>, tensor<i32>, tensor<128xf32>, tensor<128xi32>, tensor<f32>, tensor<i32>) -> tensor<?x128xi8>
  %30 = shape.shape_of %29 : tensor<?x128xi8> -> tensor<2xindex>
  %31 = "mhlo.dynamic_broadcast_in_dim"(%8, %30) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x128xf32>
  %32 = "mhlo.dynamic_broadcast_in_dim"(%9, %30) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>, tensor<2xindex>) -> tensor<?x128xi32>
  %33 = mhlo.convert %29 : (tensor<?x128xi8>) -> tensor<?x128xi32>
  %34 = mhlo.subtract %33, %32 : tensor<?x128xi32>
  %35 = mhlo.convert %34 : (tensor<?x128xi32>) -> tensor<?x128xf32>
  %36 = mhlo.multiply %35, %31 : tensor<?x128xf32>
  return %36 : tensor<?x128xf32>
}