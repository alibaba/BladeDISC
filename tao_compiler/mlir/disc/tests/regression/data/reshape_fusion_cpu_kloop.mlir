module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:1 = tf_executor.graph {
      %t0:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %0:2 = tf_executor.island wraps "tf.Neg"(%t0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %c1:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.ExpandDims"(%0, %c1) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %2:2 = tf_executor.island wraps "tf.Exp"(%1) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %3:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
      %4:2 = tf_executor.island wraps "tf.Transpose"(%2, %3) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      tf_executor.fetch %4 : tensor<?x?x?xf32>
    }
    return %graph : tensor<?x?x?xf32>
  }
}