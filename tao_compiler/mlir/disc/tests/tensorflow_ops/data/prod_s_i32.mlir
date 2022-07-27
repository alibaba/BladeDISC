module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<1xi32>) -> tensor<i32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
      %3:2 = tf_executor.island wraps "tf.Prod"(%arg0, %1) {keep_dims = false} : (tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      tf_executor.fetch %3 : tensor<i32>
    }
    return %graph : tensor<i32>
  }
}