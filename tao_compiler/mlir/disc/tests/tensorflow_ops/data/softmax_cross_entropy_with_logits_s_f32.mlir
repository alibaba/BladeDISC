func.func @main(%features: tensor<13x21xf32>, %labels: tensor<13x21xf32>) -> (tensor<13xf32>, tensor<13x21xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph:2 = tf_executor.graph {
    %0:3 = tf_executor.island wraps "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<13x21xf32>, tensor<13x21xf32>) -> (tensor<13xf32>, tensor<13x21xf32>)
    tf_executor.fetch %0#0, %0#1 : tensor<13xf32>, tensor<13x21xf32>
  }
  return %graph#0, %graph#1 : tensor<13xf32>, tensor<13x21xf32>
}