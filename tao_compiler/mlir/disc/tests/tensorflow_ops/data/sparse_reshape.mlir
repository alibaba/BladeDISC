module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xi64>, %arg1: tensor<?xi64>, %arg2: tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:3 = tf_executor.island wraps "tf.SparseReshape"(%arg0, %arg1, %arg2) : (tensor<?x?xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<?x?xi64>, tensor<?xi64>)
      %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<?x?xi64>) -> tensor<?x?xi64>
      %2:2 = tf_executor.island wraps "tf.Identity"(%0#1) : (tensor<?xi64>) -> tensor<?xi64>
      tf_executor.fetch %1#0, %2#0 : tensor<?x?xi64>, tensor<?xi64>
    }
    return %graph#0, %graph#1 : tensor<?x?xi64>, tensor<?xi64>
  }
}
