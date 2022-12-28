module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%input: tensor<1x25x2x2xf32>) -> (tensor<1x8x2x2xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %input_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<8.333333e-3> : tensor<f32>} : () -> tensor<f32>
      %input_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_input:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%input, %input_scale, %input_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<1x25x2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<1x25x2x2xf32>

      %weight:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.8> : tensor<1x1x25x8xf32>} : () -> tensor<1x1x25x8xf32>
      %weight_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.568627e-2> : tensor<8xf32>} : () -> tensor<8xf32>
      %weight_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<8xi32>} : () -> tensor<8xi32>
      %quantized_weight:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%weight, %weight_scale, %weight_zero_point) {
          axis = [3], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<1x1x25x8xf32>, tensor<8xf32>, tensor<8xi32>) -> tensor<1x1x25x8xf32>

      %result:2 = tf_executor.island wraps "tf.Conv2D"(%quantized_input, %quantized_weight)
      {
        data_format = "NCHW",
        dilations = [1, 1, 1, 1],
        padding = "SAME",
        strides = [1, 1, 1, 1],
        explicit_paddings = [0, 0, 0, 0]
      } : (tensor<1x25x2x2xf32>, tensor<1x1x25x8xf32>) -> tensor<1x8x2x2xf32>

      %result_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %result_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_result:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%result, %result_scale, %result_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<1x8x2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<1x8x2x2xf32>

      tf_executor.fetch %quantized_result : tensor<1x8x2x2xf32>
    }
    return %graph : tensor<1x8x2x2xf32>
  }
}
