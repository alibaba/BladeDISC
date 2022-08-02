module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?x!tf_type.qint8>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> (tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %3:2 = tf_executor.island wraps "tf.Dequantize"(%arg0, %arg1, %arg2) { mode = "SCALED", axis = 1, narrow_range = false } : (tensor<?x?x?x?x!tf_type.qint8>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x?x?x?xf32>)
      tf_executor.fetch %3 : tensor<?x?x?x?xf32>
    }
    return %graph : tensor<?x?x?x?xf32>
  }
}