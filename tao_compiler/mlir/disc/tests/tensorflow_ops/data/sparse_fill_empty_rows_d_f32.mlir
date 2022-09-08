module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%indices: tensor<?x?xi64>, %values: tensor<?xf32>, %dense_shape: tensor<2xi64>, %default_value: tensor<f32>) -> (tensor<?x2xi64>, tensor<?xf32>, tensor<?xi1>, tensor<?xi64>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:4 = tf_executor.graph {
      %0:5 = tf_executor.island wraps "tf.SparseFillEmptyRows"(%indices, %values, %dense_shape, %default_value) : (tensor<?x?xi64>, tensor<?xf32>, tensor<2xi64>, tensor<f32>) -> (tensor<?x2xi64>, tensor<?xf32>, tensor<?xi1>, tensor<?xi64>)
      %1:2 = tf_executor.island wraps "tf.Identity"(%0#0): (tensor<?x2xi64>) -> tensor<?x2xi64>
      %2:2 = tf_executor.island wraps "tf.Identity"(%0#1): (tensor<?xf32>) -> tensor<?xf32>
      %3:2 = tf_executor.island wraps "tf.Identity"(%0#2): (tensor<?xi1>) -> tensor<?xi1>
      %4:2 = tf_executor.island wraps "tf.Identity"(%0#3): (tensor<?xi64>) -> tensor<?xi64>
      tf_executor.fetch %1#0, %2#0, %3#0, %4#0 : tensor<?x2xi64>, tensor<?xf32>, tensor<?xi1>, tensor<?xi64>
    }
    return %graph#0, %graph#1, %graph#2, %graph#3 : tensor<?x2xi64>, tensor<?xf32>, tensor<?xi1>, tensor<?xi64>
  }
}
