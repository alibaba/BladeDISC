module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2x16xf32>, %arg1: tensor<i32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:3 = tf_executor.island wraps "tf.TopKV2"(%arg0, %arg1) {T = f32, device = "", sorted = true} : (tensor<2x16xf32>, tensor<i32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) loc("output0,output1")
      %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
      %2:2 = tf_executor.island wraps "tf.Identity"(%0#1) : (tensor<2x3xi32>) -> tensor<2x3xi32>
      tf_executor.fetch %1#0, %2#0 : tensor<2x3xf32>, tensor<2x3xi32>
    }
    return %graph#0, %graph#1 : tensor<2x3xf32>, tensor<2x3xi32>
  }
}
