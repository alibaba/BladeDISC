module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x32xf32>) -> (tensor<?x32xf32>, tensor<?x32xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.1> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
      %1:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %0) : (tensor<?x32xf32>, tensor<32x32xf32>) -> tensor<?x32xf32>
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.2> : tensor<32x32xf32>} : () -> tensor<32x32xf32>
      %3:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %2) : (tensor<?x32xf32>, tensor<32x32xf32>) -> tensor<?x32xf32>
      tf_executor.fetch %1, %3 : tensor<?x32xf32>, tensor<?x32xf32>
    }
    return %graph#0, %graph#1 : tensor<?x32xf32>, tensor<?x32xf32>
  }
}