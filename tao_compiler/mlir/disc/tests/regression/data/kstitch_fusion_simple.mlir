module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
      %2:2 = tf_executor.island wraps "tf.Max"(%arg0, %1) : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
      %3:2 = tf_executor.island wraps "tf.Neg"(%2) : (tensor<?xf32>) -> tensor<?xf32>
      tf_executor.fetch %2, %3 : tensor<?xf32>, tensor<?xf32>
    }
    return %graph#0, %graph#1 : tensor<?xf32>, tensor<?xf32>
  }
}