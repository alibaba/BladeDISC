module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {value = dense<2.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() {value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Const"() {value = dense<-2.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Const"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %outputs_10:3, %control_11 = tf_executor.island wraps "tf.QuantizeV2"(%arg0, %outputs_8, %outputs_2) {axis = -1 : i64, ensure_minimum_range = 0.00999999977 : f32, mode = "SCALED", narrow_range = true, round_mode = "HALF_AWAY_FROM_ZERO"} : (tensor<?x?x?x?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x?x?x!tf_type.qint8>, tensor<f32>, tensor<f32>)
      %outputs_12, %control_13 = tf_executor.island wraps "tf.Const"() {value = dense<-1.270000e+02> : tensor<f32>} : () -> tensor<f32> loc(fused["Const:", "min_output"])
      %outputs_14_0, %control_15_0 = tf_executor.island wraps "tf.Const"() {value = dense<-51> : tensor<1x1x25x8xi8>} : () -> tensor<1x1x25x8xi8>
      %outputs_14, %control_15 = tf_executor.island wraps "tf.Cast"(%outputs_14_0) : (tensor<1x1x25x8xi8>) -> tensor<1x1x25x8x!tf_type.qint8>
      %outputs_16:3, %control_17 = tf_executor.island wraps "tf.QuantizedConv2DWithBiasAndRequantize"(%outputs_10#0, %outputs_14, %outputs, %outputs_10#1, %outputs_10#2, %outputs_6, %outputs_0, %outputs_12, %outputs_4) {Tbias = f32, Tfilter = !tf_type.qint8, Tinput = !tf_type.qint8, dilations = [1, 1, 1, 1], out_type = !tf_type.qint8, padding = "SAME", padding_list = [], strides = [1, 1, 1, 1]} : (tensor<?x?x?x?x!tf_type.qint8>, tensor<1x1x25x8x!tf_type.qint8>, tensor<8xf32>, tensor<f32>, tensor<f32>, tensor<8xf32>, tensor<8xf32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x?x8x!tf_type.qint8>, tensor<f32>, tensor<f32>)
      %outputs_18, %control_19 = tf_executor.island wraps "tf.Dequantize"(%outputs_16#0, %outputs_16#1, %outputs_16#2) {axis = -1 : i64, mode = "SCALED", narrow_range = false} : (tensor<?x?x?x8x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> tensor<?x?x?x?xf32>
      tf_executor.fetch %outputs_18 : tensor<?x?x?x?xf32>
    }
    return %graph : tensor<?x?x?x?xf32>
  }
}