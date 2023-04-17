module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> (tensor<?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
      %1:2 = tf_executor.island wraps "tf.Mean"(%arg0, %0) : (tensor<?x?xf32>, tensor<1xi32>) -> (tensor<?xf32>)
      tf_executor.fetch %1 : tensor<?xf32>
    }
    return %graph : tensor<?xf32>
  }
}