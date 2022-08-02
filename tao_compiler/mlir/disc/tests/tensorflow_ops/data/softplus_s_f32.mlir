func.func @main(%arg0: tensor<13x21x100xf32>) -> tensor<13x21x100xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Softplus"(%arg0) : (tensor<13x21x100xf32>) -> tensor<13x21x100xf32>
    tf_executor.fetch %1 : tensor <13x21x100xf32>
  }
  return %graph : tensor<13x21x100xf32>
}