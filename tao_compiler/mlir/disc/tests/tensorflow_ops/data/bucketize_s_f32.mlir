module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2x10xf32>) -> (tensor<2x10xi32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x10xf32>) -> (tensor<2x10xi32>)
      tf_executor.fetch %0 : tensor<2x10xi32>
    }
    return %graph : tensor<2x10xi32>
  }
}
