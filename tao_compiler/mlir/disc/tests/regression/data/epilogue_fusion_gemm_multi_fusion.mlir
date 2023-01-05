// Multiple GEMMs that should be fused into different kernels.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>, %arg2: tensor<?x?x?x?xf16>) 
        -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) 
        attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      // First group of GEMM with epilogue.
      %0:2 = tf_executor.island wraps "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %1:2 = tf_executor.island wraps "tf.Abs"(%0) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %2:2 = tf_executor.island wraps "tf.Neg"(%1) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)

      // Ops that will cause multi-output if fused as epilogue of the above GEMM group.
      %3:2 = tf_executor.island wraps "tf.Rsqrt"(%2) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %shape:2 = tf_executor.island wraps "tf.Shape"(%0) {device = ""} : (tensor<?x?x?x?xf16>) -> tensor<4xi32>
      %cst:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.234> : tensor<f16>} : () -> tensor<f16>
      %cstbcast:2 = tf_executor.island wraps "tf.BroadcastTo"(%cst, %shape) : (tensor<f16>, tensor<4xi32>) -> tensor<?x?x?x?xf16>
      %4:2 = tf_executor.island wraps "tf.Mul"(%cstbcast, %2) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      %5:2 = tf_executor.island wraps "tf.Add"(%4, %4) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

      // Second group of GEMM with epilogue.
      %6:2 = tf_executor.island wraps "tf.BatchMatMul"(%5, %arg2) {adj_x = false, adj_y = false} : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %7:2 = tf_executor.island wraps "tf.Rsqrt"(%6) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %8:2 = tf_executor.island wraps "tf.Abs"(%7) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)
      %9:2 = tf_executor.island wraps "tf.Exp"(%8) : (tensor<?x?x?x?xf16>) -> (tensor<?x?x?x?xf16>)

      tf_executor.fetch %3, %9 : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
    }
    return %graph#0, %graph#1 : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }
}