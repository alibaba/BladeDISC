// This UT has an irregular xroot (i.e., `%1`).
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.ConcatV2"(%1, %arg1, %0) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<i32>) -> tensor<?x?xf32>
      %3:2 = tf_executor.island wraps "tf.Sum"(%2, %0) : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xf32>
      %4:2 = tf_executor.island wraps "tf.ConcatV2"(%1, %arg2, %0) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<i32>) -> tensor<?x?xf32>
      %5:2 = tf_executor.island wraps "tf.Add"(%2, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %6:2 = tf_executor.island wraps "tf.Sum"(%5, %0) : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xf32>
      %7:2 = tf_executor.island wraps "tf.Add"(%3, %6) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      tf_executor.fetch %1, %7 : tensor<?x?xf32>, tensor<?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?xf32>, tensor<?xf32>
  }
}