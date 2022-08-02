module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> (tensor<2x3x4x5x!tf_type.qint8>, tensor<3xf32>, tensor<3xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:3 = tf_executor.graph {
      %3:4 = tf_executor.island wraps "tf.QuantizeV2"(%arg0, %arg1, %arg2) { mode = "SCALED", axis = 1 } : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x4x5x!tf_type.qint8>, tensor<3xf32>, tensor<3xf32>)
      tf_executor.fetch %3#0, %3#1, %3#2 : tensor<2x3x4x5x!tf_type.qint8>, tensor<3xf32>, tensor<3xf32>
    }
    return %graph#0, %graph#1, %graph#2 : tensor<2x3x4x5x!tf_type.qint8>, tensor<3xf32>, tensor<3xf32>
  }
}