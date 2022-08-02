module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2, -1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %2:2 = tf_executor.island wraps "tf.Slice"(%arg0, %0, %1) {device = ""} : (tensor<3x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xf32>
      tf_executor.fetch %2 : tensor<2x3xf32>
    }
    return %graph : tensor<2x3xf32>
  }
}