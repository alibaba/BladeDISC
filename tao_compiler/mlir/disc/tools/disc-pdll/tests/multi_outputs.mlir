// RUN: disc-pdll --payload-input %s --pdl-input %p/multi_outputs.pdll  | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) {
  %w0 = "tf.Const"() {value = dense<1.0> : tensor<1x1x64x64xf32>} : () -> tensor<1x1x64x64xf32>
  // CHECK-NOT: tf.Conv2D
  // CHECK: "mhlo_disc.custom_call_v2
  // CHECK-SAME: call_target_name = "disc.custom_call.fused_conv_relu"
  // CHECK-SMAE: custom_attrs = {
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: dilation = [1, 1, 1, 1]
  // CHECK-SAME: padding = "SAME"
  // CHECK-SAME: strides = [1, 1, 1, 1]
  // CHECK-SAME: device = "h"
  // CHECK-SAME: expected_input_layouts = "NHWC,NHWC"
  // CHECK-SAME: expected_output_layouts = "NHWC,NHWC"
  // CHECK-SAME: input_layouts = "NHWC,NHWC"
  // CHECK-SAME: input_placements = "h,h"
  // CHECK-SAME: output_layouts = "NHWC,NHWC"
  // CHECK-SAME: output_placements = "h"
  %conv = "tf.Conv2D"(%arg0, %w0)
  {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    padding = "SAME",
    strides = [1, 1, 1, 1]
  } : (tensor<?x?x?x?xf32>, tensor<1x1x64x64xf32>) -> tensor<?x?x?x?xf32>
  // CHECK-NOT: tf.Relu
  %relu = "tf.Relu"(%conv) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  return %conv, %relu : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
}
