module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xi1> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
      %2:2 = tf_executor.island wraps "tf.Cast"(%arg0) : (tensor<?x?xf32>) -> (tensor<?x?xi1>)
      %3:2 = tf_executor.island wraps "tf.All"(%2, %1) {keep_dims = true} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?x?xi1>
      tf_executor.fetch %3 : tensor<?x?xi1>
    }
    return %graph : tensor<?x?xi1>
  }
}