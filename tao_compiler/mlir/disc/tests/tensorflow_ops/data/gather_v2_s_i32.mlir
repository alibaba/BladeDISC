module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<3xi32>) -> (tensor<2xi32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.Const"() { value = dense<[0,1]> : tensor<2xi32> } : () -> tensor<2xi32>
      %2:2 = tf_executor.island wraps "tf.GatherV2"(%arg0, %1, %0) : (tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
      tf_executor.fetch %2 : tensor<2xi32>
    }
    return %graph : tensor<2xi32>
  }
}