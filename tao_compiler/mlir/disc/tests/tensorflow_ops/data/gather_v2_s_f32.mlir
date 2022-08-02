module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<10x20x30xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2x3x20x30xf32>, tensor<10x2x3x30xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
      %2:2 = tf_executor.island wraps "tf.GatherV2"(%arg0, %arg1, %0) : (tensor<10x20x30xf32>, tensor<2x3xi32>, tensor<i32>) -> tensor<2x3x20x30xf32>
      %3:2 = tf_executor.island wraps "tf.GatherV2"(%arg0, %arg1, %1) : (tensor<10x20x30xf32>, tensor<2x3xi32>, tensor<i32>) -> tensor<10x2x3x30xf32>
      tf_executor.fetch %2, %3 : tensor<2x3x20x30xf32>, tensor<10x2x3x30xf32>
    }
    return %graph#0, %graph#1 : tensor<2x3x20x30xf32>, tensor<10x2x3x30xf32>
  }
}