module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2x3x110x100xf32>, %arg1: tensor<2x3x110x100xf32>) -> (tensor<2x3x100x100xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<2x3x110x100xf32>, tensor<2x3x110x100xf32>) -> (tensor<2x3x100x100xf32>)
      tf_executor.fetch %0 : tensor<2x3x100x100xf32>
    }
    return %graph : tensor<2x3x100x100xf32>
  }
}