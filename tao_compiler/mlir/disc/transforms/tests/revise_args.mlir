// RUN: disc-opt -disc-tf-revise-args-for-static-rank %s | FileCheck %s

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1x2xi64>) -> tensor<?xf32> 
    attributes {tf.entry_function = {input_placements = "cpu,const,cpu",
    inputs = "input0,input1,input2",
    output_placements = "cpu",
    outputs = "output0",
    disc.input_shape_0 = dense<0.0> : tensor<4xf32>,
    disc.input_value_1 = dense<2.0> : tensor<f32>}} {
  // CHECK: "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: "tf.PadV2"({{.*}}) : (tensor<4xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<?xf32>
  %1 = "tf.PadV2"(%arg0, %arg2, %arg1) : (tensor<?xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}