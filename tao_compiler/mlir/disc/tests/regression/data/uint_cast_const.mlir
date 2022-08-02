module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0 : tensor<?xui8>) -> (tensor<?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<ui8>} : () -> (tensor<ui8>)
      %2:2 = tf_executor.island wraps "tf.Add"(%0, %arg0) : (tensor<ui8>, tensor<?xui8>) -> (tensor<?xui8>)
      %3:2 = tf_executor.island wraps "tf.Cast"(%2) : (tensor<?xui8>) -> (tensor<?xf32>)
      tf_executor.fetch %3 : tensor<?xf32>
    }
    return %graph : tensor<?xf32>
  }
}
