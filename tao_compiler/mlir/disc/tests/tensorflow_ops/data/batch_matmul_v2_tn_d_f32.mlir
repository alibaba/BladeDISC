module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)
      tf_executor.fetch %0 : tensor<?x?x?x?xf32>
    }
    return %graph : tensor<?x?x?x?xf32>
  }
}