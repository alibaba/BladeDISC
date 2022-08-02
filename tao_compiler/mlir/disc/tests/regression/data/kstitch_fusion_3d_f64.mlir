// 3D sub-root and FP64 datatype.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xf64>) -> (tensor<?x?xf64>, tensor<?x?x?xf64>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %3:2 = tf_executor.island wraps "tf.Shape"(%arg0) {device = ""} : (tensor<?x?x?xf64>) -> (tensor<3xi32>)
      %1:2 = tf_executor.island wraps "tf.Sum"(%arg0, %0) : (tensor<?x?x?xf64>, tensor<i32>) -> tensor<?x?xf64>
      %c2:2 = tf_executor.island wraps "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
      %4:2 = tf_executor.island wraps "tf.ExpandDims"(%1, %c2) : (tensor<?x?xf64>, tensor<i32>) -> (tensor<?x?x?xf64>)
      %5:2 = tf_executor.island wraps "tf.BroadcastTo"(%4, %3) : (tensor<?x?x?xf64>, tensor<3xi32>) -> tensor<?x?x?xf64>
      %6:2 = tf_executor.island wraps "tf.Add"(%arg0, %5) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
      tf_executor.fetch %1, %6 : tensor<?x?xf64>, tensor<?x?x?xf64>
    }
    return %graph#0, %graph#1 : tensor<?x?xf64>, tensor<?x?x?xf64>
  }
}