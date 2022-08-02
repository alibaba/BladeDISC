
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<20x30xf32>) -> tensor<40x90xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() { value = dense<[2,3]> : tensor<2xi32> } : () -> tensor<2xi32>
      %2:2 = tf_executor.island wraps "tf.Tile"(%arg0, %1) : (tensor<20x30xf32>, tensor<2xi32>) -> tensor<40x90xf32>
      tf_executor.fetch %2 : tensor<40x90xf32>
    }
    return %graph : tensor<40x90xf32>
  }
}
