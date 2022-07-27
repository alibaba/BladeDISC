module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<6x7x8x9xf32>) -> tensor<6x9x8x7xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
      %2:2 = tf_executor.island wraps "tf.Transpose"(%arg0, %1) : (tensor<6x7x8x9xf32>, tensor<4xi32>) -> tensor<6x9x8x7xf32>
      tf_executor.fetch %2 : tensor<6x9x8x7xf32>
    }
    return %graph : tensor<6x9x8x7xf32>
  }
}