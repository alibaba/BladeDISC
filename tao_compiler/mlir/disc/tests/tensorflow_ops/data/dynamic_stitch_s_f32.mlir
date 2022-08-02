module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2x2x2xf32>) -> (tensor<5x2xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[[3, 2], [1, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %2:2 = tf_executor.island wraps "tf.DynamicStitch"(%0, %1, %arg0, %arg1) {} : (tensor<i32>, tensor<2x2xi32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<5x2xf32>
      tf_executor.fetch %2 : tensor<5x2xf32>
    }
    return %graph : tensor<5x2xf32>
  }
}
