module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2x1x3xf32>) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Squeeze"(%arg0) {T = f32, device = "", squeeze_dims = [1]} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
      tf_executor.fetch %1 : tensor <2x3xf32>
    }
    return %graph : tensor <2x3xf32>
  }
}
