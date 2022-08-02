module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x3x?x?x!tf_type.qint8>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<?x3x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %3:2 = tf_executor.island wraps "tf.Dequantize"(%arg0, %arg1, %arg2) { mode = "SCALED", axis = -1, narrow_range = false } : (tensor<?x3x?x?x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> (tensor<?x3x?x?xf32>)
      tf_executor.fetch %3 : tensor<?x3x?x?xf32>
    }
    return %graph : tensor<?x3x?x?xf32>
  }
}