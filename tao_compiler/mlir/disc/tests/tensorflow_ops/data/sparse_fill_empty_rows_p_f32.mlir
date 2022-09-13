module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%indices: tensor<?x2xi64>, %values: tensor<?xf32>, %default_value: tensor<f32>) -> (tensor<?x2xi64>, tensor<?xf32>, tensor<5xi1>, tensor<?xi64>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:4 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[5, 6]> : tensor<2xi64>} : () -> tensor<2xi64>
      %1:5 = tf_executor.island wraps "tf.SparseFillEmptyRows"(%indices, %values, %0, %default_value) : (tensor<?x2xi64>, tensor<?xf32>, tensor<2xi64>, tensor<f32>) -> (tensor<?x2xi64>, tensor<?xf32>, tensor<5xi1>, tensor<?xi64>)
      %2:2 = tf_executor.island wraps "tf.Identity"(%1#0): (tensor<?x2xi64>) -> tensor<?x2xi64>
      %3:2 = tf_executor.island wraps "tf.Identity"(%1#1): (tensor<?xf32>) -> tensor<?xf32>
      %4:2 = tf_executor.island wraps "tf.Identity"(%1#2): (tensor<5xi1>) -> tensor<5xi1>
      %5:2 = tf_executor.island wraps "tf.Identity"(%1#3): (tensor<?xi64>) -> tensor<?xi64>
      tf_executor.fetch %2#0, %3#0, %4#0, %5#0 : tensor<?x2xi64>, tensor<?xf32>, tensor<5xi1>, tensor<?xi64>
    }
    return %graph#0, %graph#1, %graph#2, %graph#3 : tensor<?x2xi64>, tensor<?xf32>, tensor<5xi1>, tensor<?xi64>
  }
}
