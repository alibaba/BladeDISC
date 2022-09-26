module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%input: tensor<?x?xf32>) -> (tensor<?x71xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %input_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<8.333333e-3> : tensor<f32>} : () -> tensor<f32>
      %input_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_input:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%input, %input_scale, %input_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<?x?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x?xf32>

      %weight:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.8> : tensor<25x71xf32>} : () -> tensor<25x71xf32>
      %weight_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.568627e-2> : tensor<71xf32>} : () -> tensor<71xf32>
      %weight_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<71xi32>} : () -> tensor<71xi32>
      %quantized_weight:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%weight, %weight_scale, %weight_zero_point) {
          axis = [1], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<25x71xf32>, tensor<71xf32>, tensor<71xi32>) -> tensor<25x71xf32>

      %result:2 = tf_executor.island wraps "tf.MatMul"(%quantized_input, %quantized_weight)
      {
        transpose_a = false,
        transpose_b = false
      } : (tensor<?x?xf32>, tensor<25x71xf32>) -> tensor<?x71xf32>

      %result_scale:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %result_zero_point:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %quantized_result:2 = tf_executor.island wraps "tf.DiscFakeQuant"(%result, %result_scale, %result_zero_point) {
          axis = [], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} :
      (tensor<?x71xf32>, tensor<f32>, tensor<i32>) -> tensor<?x71xf32>

      tf_executor.fetch %quantized_result : tensor<?x71xf32>
    }
    return %graph : tensor<?x71xf32>
  }
}