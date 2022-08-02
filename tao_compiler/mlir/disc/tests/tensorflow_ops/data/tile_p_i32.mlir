module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<?xi32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %2:2 = tf_executor.island wraps "tf.Tile"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
      tf_executor.fetch %2 : tensor<?xi32>
    }
    return %graph : tensor<?xi32>
  }
}