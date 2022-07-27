module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<i1> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.RandomUniform"(%arg0) { seed = 2, seed2 = 3} : (tensor<2xi32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Sub"(%1, %arg1) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
      %3:2 = tf_executor.island wraps "tf.Mul"(%2, %2) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0,1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %5:2 = tf_executor.island wraps "tf.Mean"(%3, %4) : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
      %6:2 = tf_executor.island wraps "tf.Sub"(%5, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %7:2 = tf_executor.island wraps "tf.Abs"(%6) : (tensor<f32>) -> tensor<f32>
      %8:2 = tf_executor.island wraps "tf.LessEqual"(%7, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<i1>
      tf_executor.fetch %8 : tensor<i1>
    }
    return %graph : tensor<i1>
  }
}