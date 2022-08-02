module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:1 = tf_executor.graph {
      // Stable softmax with preprocessing: z = x - max(x)
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %max:2 = tf_executor.island wraps "tf.Max"(%arg0, %0) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %c2:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %max_e:2 = tf_executor.island wraps "tf.ExpandDims"(%max, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %s:2 = tf_executor.island wraps "tf.Shape"(%arg0) {device = ""} : (tensor<?x?x?xf32>) -> (tensor<3xi32>)
      %max_f:2 = tf_executor.island wraps "tf.BroadcastTo"(%max_e, %s) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %z:2 = tf_executor.island wraps "tf.Sub"(%arg0, %max_f) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      // exp(z)
      %1:2 = tf_executor.island wraps "tf.Exp"(%z) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      // sum(exp(z))
      %3:2 = tf_executor.island wraps "tf.Sum"(%1, %0) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.ExpandDims"(%3, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %5:2 = tf_executor.island wraps "tf.BroadcastTo"(%4, %s) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      // softmax = exp(z) / sum(exp(z))
      %6:2 = tf_executor.island wraps "tf.Div"(%1, %5) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      tf_executor.fetch %6 : tensor<?x?x?xf32>
    }
    return %graph : tensor<?x?x?xf32>
  }
}