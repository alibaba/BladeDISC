module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<i1> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.RandomUniform"(%arg0) { seed = 1, seed2 = 0} : (tensor<2xi32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0,1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %3:2 = tf_executor.island wraps "tf.Mean"(%1, %2) : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
      %4:2 = tf_executor.island wraps "tf.Sub"(%3, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %5:2 = tf_executor.island wraps "tf.Abs"(%4) : (tensor<f32>) -> tensor<f32>
      %6:2 = tf_executor.island wraps "tf.LessEqual"(%5, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<i1>
      tf_executor.fetch %6 : tensor<i1>
    }
    return %graph : tensor<i1>
  }
}