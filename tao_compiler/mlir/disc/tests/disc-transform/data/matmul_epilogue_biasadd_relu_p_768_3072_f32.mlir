module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.8> : tensor<768x3072xf32>} : () -> tensor<768x3072xf32>
      %1:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %0) {transpose_a = false, transpose_b = false} : (tensor<?x?xf32>, tensor<768x3072xf32>) -> (tensor<?x?xf32>)
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.1> : tensor<3072xf32>} : () -> tensor<3072xf32>
      %3:2 = tf_executor.island wraps "tf.BiasAdd"(%1, %2) : (tensor<?x?xf32>, tensor<3072xf32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.Relu"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %4 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}