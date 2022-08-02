module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %cst:2 = tf_executor.island wraps "tf.Const"() {value = dense<[[0, 0], [1, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %1:2 = tf_executor.island wraps "tf.Pad"(%arg0, %cst) : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
      tf_executor.fetch %1 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}
