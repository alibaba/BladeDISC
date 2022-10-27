module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?x?x?xi1>) -> tensor<?x?xi64> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Where"(%arg0) : (tensor<?x?x?x?x?x?xi1>) -> tensor<?x?xi64>
      %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<?x?xi64>) -> tensor<?x?xi64>
      tf_executor.fetch %1#0 : tensor<?x?xi64>
    }
    return %graph : tensor<?x?xi64>
  }
}
