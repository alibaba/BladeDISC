module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<100x100xf32>) -> tensor<1x1xi1> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0,1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %2:2 = tf_executor.island wraps "tf.Cast"(%arg0) : (tensor<100x100xf32>) -> (tensor<100x100xi1>)
      %3:2 = tf_executor.island wraps "tf.All"(%2, %1) {keep_dims = true} : (tensor<100x100xi1>, tensor<2xi32>) -> tensor<1x1xi1>
      tf_executor.fetch %3 : tensor<1x1xi1>
    }
    return %graph : tensor<1x1xi1>
  }
}