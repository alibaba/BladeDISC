func.func @main(%arg0: tensor<100x100x3x2xf32>) -> (tensor<100x100x3xf32>, tensor<100x100x3xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph:2 = tf_executor.graph {
    %0:3 = tf_executor.island wraps "tf.Unpack"(%arg0) {T = f32, axis = -1 : i64} : (tensor<100x100x3x2xf32>) -> (tensor<100x100x3xf32>, tensor<100x100x3xf32>)
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<100x100x3xf32>) -> tensor<100x100x3xf32>
    %2:2 = tf_executor.island wraps "tf.Identity"(%0#1) : (tensor<100x100x3xf32>) -> tensor<100x100x3xf32>
    tf_executor.fetch %1#0, %2#0 : tensor<100x100x3xf32>, tensor<100x100x3xf32>
  }
  return %graph#0, %graph#1 : tensor<100x100x3xf32>, tensor<100x100x3xf32>
}
