module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> (tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
      %3:2 = tf_executor.island wraps "tf.StridedSlice"(%arg0, %arg1, %arg2, %2) {
          begin_mask = 9 : i64, ellipsis_mask = 0 : i64, end_mask = 9 : i64, new_axis_mask = 6 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x?xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
      tf_executor.fetch %3 : tensor<?x?x?x?xf32>
    }
    return %graph: tensor<?x?x?x?xf32>
  }
}