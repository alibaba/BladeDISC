module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i64, producer = 0 : i64}} {
  func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xi64>) -> (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
      %1:2 = tf_executor.island wraps "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
      %2:2 = tf_executor.island wraps "tf.GatherV2"(%arg0, %arg1, %0) : (tensor<?x?x?xf32>, tensor<?x?xi64>, tensor<i64>) -> tensor<?x?x?x?xf32>
      %3:2 = tf_executor.island wraps "tf.GatherV2"(%arg0, %arg1, %1) : (tensor<?x?x?xf32>, tensor<?x?xi64>, tensor<i64>) -> tensor<?x?x?x?xf32>
      tf_executor.fetch %2, %3 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
  }
}