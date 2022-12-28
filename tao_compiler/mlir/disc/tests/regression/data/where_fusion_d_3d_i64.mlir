module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 561 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xi64>) -> tensor<?x3xi64> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %0 = tf_executor.graph {
      %c_0:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i64>} : () -> tensor<i64>
      %1:2 = tf_executor.island wraps "tf.NotEqual"(%arg0, %c_0) {device = "", incompatible_shape_error = true} : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi1>
      %2:2 = tf_executor.island wraps "tf.Where"(%1) {device = ""} : (tensor<?x?x?xi1>) -> tensor<?x3xi64>
      tf_executor.fetch %2: tensor<?x3xi64>
    }
    return %0 : tensor<?x3xi64>
  }
}
