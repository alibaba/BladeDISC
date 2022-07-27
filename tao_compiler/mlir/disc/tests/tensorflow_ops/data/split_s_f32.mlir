module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<100x100xf32>) -> (tensor<100x50xf32>, tensor<100x50xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %num_split:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %2:3 = tf_executor.island wraps "tf.Split"(%num_split, %arg0) {T = i32, num_split = 2 : i64} : (tensor<i32>, tensor<100x100xf32>) -> (tensor<100x50xf32>, tensor<100x50xf32>)
      %3:2 = tf_executor.island wraps "tf.Identity"(%2#0) : (tensor<100x50xf32>) -> tensor<100x50xf32>
      %4:2 = tf_executor.island wraps "tf.Identity"(%2#1) : (tensor<100x50xf32>) -> tensor<100x50xf32>
      tf_executor.fetch %3#0, %4#0 : tensor<100x50xf32>, tensor<100x50xf32>
    }
    return %graph#0, %graph#1 : tensor<100x50xf32>, tensor<100x50xf32>
  }
}