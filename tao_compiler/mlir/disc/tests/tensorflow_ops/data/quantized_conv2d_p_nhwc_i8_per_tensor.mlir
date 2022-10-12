module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%input: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x16xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %input_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
      %input_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_input:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%input, %input_scale, %input_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<?x?x?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xf32>

      %weight:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<1x1x16x16xf32>} : () -> tensor<1x1x16x16xf32>
      %weight_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<16xf32>} : () -> tensor<16xf32>
      %weight_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<16xi32>} : () -> tensor<16xi32>
      %quantized_weight:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%weight, %weight_scale, %weight_zero_point) {
          axis = [3], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<1x1x16x16xf32>, tensor<16xf32>, tensor<16xi32>) -> tensor<1x1x16x16xf32>

      %result:2 = tf_executor.island wraps "tf.Conv2D"(%quantized_input, %quantized_weight)
      {
        data_format = "NHWC",
        dilations = [1, 1, 1, 1],
        padding = "SAME",
        strides = [1, 1, 1, 1]
      } : (tensor<?x?x?x?xf32>, tensor<1x1x16x16xf32>) -> tensor<?x?x?x16xf32>

      %result_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %result_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_result:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%result, %result_scale, %result_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<?x?x?x16xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?x?x16xf32>

      tf_executor.fetch %quantized_result : tensor<?x?x?x16xf32>
    }
    return %graph : tensor<?x?x?x16xf32>
  }
}