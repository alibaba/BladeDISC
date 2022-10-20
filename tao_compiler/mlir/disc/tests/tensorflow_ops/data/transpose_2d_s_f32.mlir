module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<6x7xf32>) -> tensor<7x6xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
      %2:2 = tf_executor.island wraps "tf.Transpose"(%arg0, %1) : (tensor<6x7xf32>, tensor<2xi32>) -> tensor<7x6xf32>
      tf_executor.fetch %2 : tensor<7x6xf32>
    }
    return %graph : tensor<7x6xf32>
  }
}