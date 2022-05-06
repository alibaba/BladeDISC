module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
      tf_executor.fetch %0 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}