module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:1 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      // exp(x)
      %1:2 = tf_executor.island wraps "tf.Exp"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      // sum(exp(x))
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %3:2 = tf_executor.island wraps "tf.Sum"(%1, %2) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %c2:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %s:2 = tf_executor.island wraps "tf.Shape"(%1) {device = ""} : (tensor<?x?x?xf32>) -> (tensor<3xi32>)
      %4:2 = tf_executor.island wraps "tf.ExpandDims"(%3, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %5:2 = tf_executor.island wraps "tf.BroadcastTo"(%4, %s) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      // softmax = exp(x) / sum(exp(x))
      %6:2 = tf_executor.island wraps "tf.Div"(%1, %5) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      tf_executor.fetch %4 : tensor<?x?x?xf32>
    }
    return %graph : tensor<?x?x?xf32>
  }
}