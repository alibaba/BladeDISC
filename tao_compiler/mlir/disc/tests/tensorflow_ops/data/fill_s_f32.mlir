module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<f32>) -> (tensor<3x5xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %cst:2 = tf_executor.island wraps "tf.Const"() { value = dense<[3,5]> : tensor<2xi32> } : () -> tensor<2xi32>
      %0:2 = tf_executor.island wraps "tf.Fill"(%cst, %arg0) : (tensor<2xi32>, tensor<f32>) -> (tensor<3x5xf32>)
      tf_executor.fetch %0 : tensor<3x5xf32>
    }
    return %graph : tensor<3x5xf32>
  }
}