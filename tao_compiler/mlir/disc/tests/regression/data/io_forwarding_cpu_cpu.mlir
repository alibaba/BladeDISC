module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Identity"(%arg0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %1:2 = tf_executor.island wraps "tf.Identity"(%arg1) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %1, %0 : tensor<?x?xf32>, tensor<?x?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
}