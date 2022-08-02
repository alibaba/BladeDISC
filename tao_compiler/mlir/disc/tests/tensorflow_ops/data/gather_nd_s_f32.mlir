module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<7x8x90xf32>, %arg1: tensor<2x3x2xi32>) -> tensor<2x3x90xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %2:2 = tf_executor.island wraps "tf.GatherNd"(%arg0, %arg1) : (tensor<7x8x90xf32>, tensor<2x3x2xi32>) -> tensor<2x3x90xf32>
      tf_executor.fetch %2 : tensor<2x3x90xf32>
    }
    return %graph : tensor<2x3x90xf32>
  }
}