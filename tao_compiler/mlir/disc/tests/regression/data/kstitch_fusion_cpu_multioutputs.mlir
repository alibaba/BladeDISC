module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %1:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %11:2 = tf_executor.island wraps "tf.Exp"(%1) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %3:2 = tf_executor.island wraps "tf.Sum"(%11, %2) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %c2:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %s:2 = tf_executor.island wraps "tf.Shape"(%1) {device = ""} : (tensor<?x?x?xf32>) -> (tensor<3xi32>)
      %4:2 = tf_executor.island wraps "tf.ExpandDims"(%3, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %5:2 = tf_executor.island wraps "tf.BroadcastTo"(%4, %s) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %6:2 = tf_executor.island wraps "tf.Div"(%1, %5) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      tf_executor.fetch %1, %6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
  }
}