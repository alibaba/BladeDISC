module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> (tensor<3x2x!tf_type.qint8>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[[0,1],[2,3],[4,5]]> : tensor<3x2xi8>} : () -> (tensor<3x2xi8>)
      %1:2 = tf_executor.island wraps "tf.Cast"(%0) : (tensor<3x2xi8>) -> tensor<3x2x!tf_type.qint8>
      tf_executor.fetch %1 : tensor<3x2x!tf_type.qint8>
    }
    return %graph : tensor<3x2x!tf_type.qint8>
  }
}