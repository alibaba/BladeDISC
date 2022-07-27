module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<1x3x100x110xf32>, %arg1: tensor<2x1x110x100xf32>) -> (tensor<2x3x100x100xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x3x100x110xf32>, tensor<2x1x110x100xf32>) -> (tensor<2x3x100x100xf32>)
      tf_executor.fetch %0 : tensor<2x3x100x100xf32>
    }
    return %graph : tensor<2x3x100x100xf32>
  }
}