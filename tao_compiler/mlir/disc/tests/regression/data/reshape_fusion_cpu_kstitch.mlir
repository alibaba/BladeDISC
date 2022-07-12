module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x1xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %1:2 = tf_executor.island wraps "tf.Exp"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %3:2 = tf_executor.island wraps "tf.ExpandDims"(%1, %2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %4:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %5:2 = tf_executor.island wraps "tf.Mean"(%3, %4) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x1xf32>
      tf_executor.fetch %0, %5 : tensor<?x?xf32>, tensor<?x1xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?xf32>, tensor<?x1xf32>
  }
}