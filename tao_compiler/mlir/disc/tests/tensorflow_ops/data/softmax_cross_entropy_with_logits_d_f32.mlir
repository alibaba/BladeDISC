func.func @main(%features: tensor<?x?xf32>, %labels: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph:2 = tf_executor.graph {
    %0:3 = tf_executor.island wraps "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>)
    tf_executor.fetch %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
  }
  return %graph#0, %graph#1 : tensor<?xf32>, tensor<?x?xf32>
}