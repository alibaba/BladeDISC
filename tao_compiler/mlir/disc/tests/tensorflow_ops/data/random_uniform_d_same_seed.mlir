module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<i1>, tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.RandomUniform"(%arg0) { seed = 2, seed2 = 3} : (tensor<2xi32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.RandomUniform"(%arg0) { seed = 3, seed2 = 2} : (tensor<2xi32>) -> tensor<?x?xf32>
      %3:2 = tf_executor.island wraps "tf.RandomUniform"(%arg0) { seed = 2, seed2 = 3} : (tensor<2xi32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.Sub"(%3, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %5:2 = tf_executor.island wraps "tf.Mul"(%1, %2) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %6:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0,1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %7:2 = tf_executor.island wraps "tf.Mean"(%5, %6) : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
      %8:2 = tf_executor.island wraps "tf.Sub"(%7, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %9:2 = tf_executor.island wraps "tf.Abs"(%8) : (tensor<f32>) -> tensor<f32>
      %10:2 = tf_executor.island wraps "tf.LessEqual"(%9, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<i1>
      tf_executor.fetch %10, %4 : tensor<i1>, tensor<?x?xf32>
    }
    return %graph#0, %graph#1 : tensor<i1>, tensor<?x?xf32>
  }
}
