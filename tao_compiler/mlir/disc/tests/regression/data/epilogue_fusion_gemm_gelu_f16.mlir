// GEMM + gelu
// gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      // rsqrt(2) = 0.7071068
      %shape:2 = tf_executor.island wraps "tf.Shape"(%0) {device = ""} : (tensor<?x?x?x?xf16>) -> tensor<4xi32>
      %cst:2 = tf_executor.island wraps "tf.Const"() {value = dense<7.071068e-1> : tensor<f16>} : () -> tensor<f16>
      %cstbcast:2 = tf_executor.island wraps "tf.BroadcastTo"(%cst, %shape) : (tensor<f16>, tensor<4xi32>) -> tensor<?x?x?x?xf16>
      // x * rsqrt(2)
      %1:2 = tf_executor.island wraps "tf.Mul"(%0, %cstbcast) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      // erf(...)
      %2:2 = tf_executor.island wraps "tf.Erf"(%1) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      // 1 + erf(...)
      %cst1:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0e+00> : tensor<f16>} : () -> tensor<f16>
      %cst1bcast:2 = tf_executor.island wraps  "tf.BroadcastTo"(%cst1, %shape) : (tensor<f16>, tensor<4xi32>) -> tensor<?x?x?x?xf16>
      %3:2 = tf_executor.island wraps "tf.Add"(%cst1bcast, %2) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      %csthalf:2 = tf_executor.island wraps "tf.Const"() {value = dense<5.0e-1> : tensor<f16>} : () -> tensor<f16>
      %csthalfbcast:2 = tf_executor.island wraps "tf.BroadcastTo"(%csthalf, %shape) : (tensor<f16>, tensor<4xi32>) -> tensor<?x?x?x?xf16>
      // 1/2 * ...
      %4:2 = tf_executor.island wraps "tf.Mul"(%csthalfbcast, %3) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      // x * ...
      %5:2 = tf_executor.island wraps "tf.Mul"(%0, %4) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

      tf_executor.fetch %5 : tensor<?x?x?x?xf16>
    }
    return %graph : tensor<?x?x?x?xf16>
  }
}