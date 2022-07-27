module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>) -> (tensor<?x?xf16>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>)
      tf_executor.fetch %0 : tensor<?x?xf16>
    }
    return %graph : tensor<?x?xf16>
  }
}