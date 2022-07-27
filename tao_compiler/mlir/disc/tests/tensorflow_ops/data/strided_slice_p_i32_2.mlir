module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?xi32>) -> (tensor<i32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %3:2 = tf_executor.island wraps "tf.StridedSlice"(%arg0, %2, %1, %2) {
          begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      tf_executor.fetch %3 : tensor<i32>
    }
    return %graph: tensor<i32>
  }
}