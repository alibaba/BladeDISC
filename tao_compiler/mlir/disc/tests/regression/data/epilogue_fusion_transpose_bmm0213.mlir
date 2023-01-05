// GEMM + BMM0213 transpose
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x16x128x768xf16>, %arg1: tensor<?x16x768x768xf16>) -> (tensor<?x128x16x768xf16>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<?x16x128x768xf16>, tensor<?x16x768x768xf16>) -> (tensor<?x16x128x768xf16>)
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
      %2:2 = tf_executor.island wraps "tf.Transpose"(%0, %1) : (tensor<?x16x128x768xf16>, tensor<4xi32>) -> tensor<?x128x16x768xf16>

      tf_executor.fetch %2 : tensor<?x128x16x768xf16>
    }
    return %graph : tensor<?x128x16x768xf16>
  }
}